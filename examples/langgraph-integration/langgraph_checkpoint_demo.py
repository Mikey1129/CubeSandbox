from __future__ import annotations

import itertools
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from cubesandbox import Sandbox

load_dotenv()

for v in ("CUBE_TEMPLATE_ID",):
    if not os.environ.get(v):
        raise SystemExit(f"Missing env: {v}")

_llm_key = os.getenv("OPENAI_API_KEY") or os.getenv("TOKENHUB_API_KEY")
if not _llm_key:
    raise SystemExit("Missing LLM API key")

llm = ChatOpenAI(
    model=os.getenv("CHAT_MODEL") or "deepseek-v3",
    api_key=_llm_key,
    base_url=os.getenv("OPENAI_BASE_URL") or "https://tokenhub.tencentmaas.com/v1",
    timeout=60, max_retries=2, temperature=0,
)


class AgentState(TypedDict):
    """State shared across nodes. `messages` accumulates the conversation;
    `attempts` / `done` drive the retry loop. The sandbox itself is shared
    through the `run_python` closure, not through `State`."""
    messages: Annotated[list, add_messages]
    attempts: int
    done: bool


def make_run_python(sandbox: Sandbox):
    """Return a `run_python` tool bound to one Cube sandbox — the same
    `cubesandbox` SDK pattern used by the LangChain guide."""
    _counter = itertools.count()

    def run_python(code: str) -> str:
        script = f"/workspace/_agent_{next(_counter)}.py"
        try:
            sandbox.files.write(script, code)
            result = sandbox.commands.run(f"python3 {script}", timeout=120, cwd="/workspace")
        except Exception as exc:
            # A dead sandbox, a failed write, transport errors and per-command
            # timeouts all raise here; surface them as tool output so the reviewer
            # can see the failure and decide to retry, rather than letting the
            # exception abort the whole graph run.
            return f"[command error] {exc}"
        out = result.stdout
        if result.stderr:
            out += "\n--- stderr ---\n" + result.stderr
        if result.exit_code != 0:
            out += f"\n[non-zero exit code: {result.exit_code}]"
        return out

    return run_python


CODER_PROMPT = (
    "You are a data analyst. Write a single self-contained Python script that answers "
    "the latest user request using the dataset at /workspace/sales.csv "
    "(columns month,product,units,price). The environment has pandas, numpy, "
    "matplotlib, scikit-learn preinstalled. Print the final numbers. Do not rely on "
    "network access. Wrap the script in a single markdown ```python ... ``` fenced "
    "block. If a previous reviewer message said RETRY, fix the issues it "
    "listed before re-running."
)

REVIEWER_PROMPT = (
    "You are a reviewer. Given the user request and the latest code output, decide "
    "whether the request is fully answered. Reply starting with exactly one word, "
    "`DONE` or `RETRY`, optionally followed by one line explaining what is missing."
)


def extract_text(content) -> str:
    """Return plain text from a message content, which may be a str, a single
    content block dict, or a list of blocks (some OpenAI-compatible endpoints
    return the latter two)."""
    if not content:
        return ""                     # None / empty content — treat as no text
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]           # a lone content block, not a list
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
        return "\n".join(parts)
    return str(content)


def strip_code_fence(text: str) -> str | None:
    """Return the code inside the first markdown fence, or None when the reply
    has no fenced block — models often prefix the fenced block with prose."""
    fence = "`" * 3                      # three backticks, without a literal fence
    if not text:
        return None
    # The opener may share a line with prose (`` The code is: ```python ``), so
    # search for the first fence token anywhere rather than only at a line start.
    start = text.find(fence)
    if start == -1:
        return None
    # Code begins on the line after the opener; a language tag like ```python and
    # any trailing prose on that line are dropped.
    after = text[start + len(fence):]
    first_nl = after.find("\n")
    if first_nl == -1:
        return None                      # opener with no body — treat as no code
    inner = []
    for line in after[first_nl + 1:].splitlines():
        # A closer starts with a fence run of three-or-more backticks and is
        # otherwise empty (a bare ``` line) or followed only by prose on the same
        # line — models sometimes slip as `` ``` Done! ``. A language-tagged line
        # like ```markdown inside the script stays code. (Known limit: a bare ```
        # line inside a docstring would still close early.)
        s = line.strip()
        if s.startswith(fence) and set(s.split(None, 1)[0]) == {"`"}:
            break
        inner.append(line)
    return "\n".join(inner).strip() or None  # empty fence (```\n```) counts as no code


def coder(state: AgentState, run_python) -> dict:
    """Ask the LLM for code, execute it in the Cube sandbox, append the output."""
    code = strip_code_fence(extract_text(llm.invoke(
        [{"role": "system", "content": CODER_PROMPT}, *state["messages"]]
    ).content))
    if code is None:
        # The model returned no fenced code block; surface that instead of writing
        # prose to a .py file and wasting an attempt on a SyntaxError.
        return {"messages": [{"role": "assistant",
                              "content": "[code output]\n(no code block in model reply)"}]}
    output = run_python(code)
    # Cap both the code and the output so a huge result (e.g. a printed DataFrame)
    # or a large generated script cannot blow the model's context window across
    # retries. Keep the tail of each: the printed numbers (and any stderr/traceback)
    # land at the end, and the script's tail still shows the logic that ran.
    if len(code) > 4000:
        code = "[earlier code truncated]\n" + code[-4000:]
    if len(output) > 4000:
        output = "[earlier output truncated]\n" + output[-4000:]
    # Include the code alongside the output so that on a RETRY the coder can see
    # what it wrote last time instead of repeating the same mistake.
    return {"messages": [{"role": "assistant",
                          "content": f"[code]\n{code}\n[code output]\n{output}"}]}


def reviewer(state: AgentState) -> dict:
    """Judge whether the latest output answers the request."""
    verdict = extract_text(llm.invoke(
        [{"role": "system", "content": REVIEWER_PROMPT}, *state["messages"]]
    ).content).strip().upper()
    # Treat the full token list as authoritative: an explicit RETRY anywhere in
    # the reply wins over DONE — the prompt asks for a single leading verdict
    # word, so a stray "DONE" inside a RETRY explanation must not stop the loop.
    # DONE only wins when no RETRY token is present.
    tokens = [t.strip(":*#`").rstrip(".,!?;") for t in verdict.split()]
    done = "DONE" in tokens and "RETRY" not in tokens
    # Emit the verdict as a user-role message so the coder treats RETRY as a
    # directive to fix, not as its own prior assistant output.
    return {
        "messages": [{"role": "user", "content": f"[reviewer] {verdict}"}],
        "attempts": state.get("attempts", 0) + 1,
        "done": done,
    }


def route_after_review(state: AgentState) -> str:
    """Conditional edge: retry `coder`, or finish after N attempts."""
    if state["done"] or state["attempts"] >= 3:
        return "end"
    return "retry"


def build_graph(run_python, checkpointer=None):
    builder = StateGraph(AgentState)
    builder.add_node("coder", lambda s: coder(s, run_python))
    builder.add_node("reviewer", reviewer)
    builder.add_edge(START, "coder")
    builder.add_edge("coder", "reviewer")
    builder.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"retry": "coder", "end": END},
    )
    return builder.compile(checkpointer=checkpointer)


def stage_input(messages):
    """Build the graph input for one stage. `attempts`/`done` have no reducer,
    so explicit values here overwrite whatever the checkpoint stored. `messages`
    uses `add_messages`, so new messages are APPENDED to the prior history —
    use a fresh thread_id if you want a clean slate."""
    return {"messages": messages, "attempts": 0, "done": False}


if __name__ == "__main__":
    checkpointer = MemorySaver()  # in-process demo; use a durable checkpointer for real resume

    sandbox = Sandbox.create(template=os.environ["CUBE_TEMPLATE_ID"], timeout=1800)
    # Key the checkpoint thread by the sandbox id so the same thread_id reattaches
    # to the same MicroVM across pause() / connect().
    config = {"configurable": {"thread_id": sandbox.sandbox_id}}
    try:
        graph = build_graph(make_run_python(sandbox), checkpointer=checkpointer)
        graph.invoke(stage_input([{"role": "user", "content": "first task"}]), config=config)

        sandbox.pause()                                   # snapshot VM + rootfs
        # Sandbox.connect() returns a NEW instance; the run_python closure captured
        # the pre-pause one, so rebind the tool and rebuild the graph on the new
        # instance, keeping the same checkpointer so the same checkpoint thread resumes.
        sandbox = Sandbox.connect(sandbox.sandbox_id)     # /workspace intact after resume
        graph = build_graph(make_run_python(sandbox), checkpointer=checkpointer)
        result = graph.invoke(stage_input([{"role": "user", "content": "follow-up task"}]), config=config)
        # Print the second stage's code output so the resume is observable.
        for msg in reversed(result["messages"]):
            content = str(msg.content)
            marker = "[code output]"
            idx = content.find(marker)
            if idx != -1:
                print(content[idx:])
                break
        else:
            print("(no code output)")
        if not result["done"]:
            print("\n(not verified: reviewer never returned DONE)")
    finally:
        sandbox.kill()
