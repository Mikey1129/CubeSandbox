---
title: LangGraph Integration Guide
author: Mikey1129
date: 2026-08-29
tags:
  - integration
  - langgraph
  - agent
lang: en-US
---

# LangGraph Integration Guide

Run a [LangGraph](https://github.com/langchain-ai/langgraph) agent — an explicit graph of nodes
and conditional edges — that executes Python inside a
[CubeSandbox](https://github.com/TencentCloud/CubeSandbox) MicroVM. Because Cube exposes an
**E2B-compatible API**, the code-execution tool is a drop-in swap from E2B to Cube, while every line
of agent-generated code gets KVM-level isolation.

This is the LangGraph counterpart to the [LangChain integration](./langchain.md). Where the
LangChain guide uses the high-level `create_agent` helper, this guide builds the graph **explicitly**
with `StateGraph`, so you control the control flow, share one sandbox across nodes, and can wire
LangGraph checkpointing to Cube's `pause()` / `connect()`.

## LangGraph vs `create_agent`

| | `create_agent` (LangChain guide) | Explicit `StateGraph` (this guide) |
|---|---|---|
| Graph shape | Fixed tool-calling loop, hidden from you | You define every node and edge |
| Retry / loops | Implicit in the agent loop | Explicit `add_conditional_edges` |
| State | Opaque message history | Typed `State` you design (sandbox id, attempts, verdict…) |
| Resume | Not directly exposed | `checkpointer` ↔ Cube `pause()` / `connect()` |

Use `create_agent` when you just need a tool-calling agent. Reach for an explicit `StateGraph` when
you want a multi-step workflow — generate → execute → review → retry — with the sandbox shared across
stages and the run resumable mid-graph.

## Components and versions

| Component | Version | Notes |
|---|---|---|
| langgraph | `>=0.2` | `StateGraph`, `START`/`END`, `add_messages` |
| langchain-openai | `>=1.0,<2.0` | `ChatOpenAI` (any OpenAI-compatible endpoint) |
| cubesandbox SDK | `>=0.6.0` | `Sandbox.create` / `files.write` / `commands.run` |
| CubeSandbox platform | `>=0.3.0` | core; higher for optional features (see LangChain guide) |
| CubeSandbox base image | `ghcr.io/tencentcloud/cubesandbox-base:2026.16` | layer your Python stack on top |

## Prerequisites

The prerequisites are identical to the LangChain guide — the same sandbox template works for both:

- CubeSandbox deployed, CubeAPI reachable at `http://<node>:3000`.
- A template image with the Python stack (pandas / numpy / matplotlib / scikit-learn) built and
  registered. Follow the LangChain guide's
  [template image](../integrations/langchain.md) steps, or reuse the same template id.
- `cubesandbox` SDK env vars: `CUBE_API_URL`, `CUBE_TEMPLATE_ID`, `CUBE_PROXY_NODE_IP`.
- An OpenAI-compatible LLM endpoint via `OPENAI_BASE_URL` / `OPENAI_API_KEY`.

## Integration steps

### 1. Build the template image

Reuse the LangChain guide's `Dockerfile` (a Python data-science stack layered on `cubesandbox-base`,
with envd listening on `:49983`). No LangGraph-specific packages need to be baked into the image —
the graph runs on the host and only *code execution* happens inside the sandbox.

### 2. Register the template and configure env vars

```bash
cubemastercli tpl create-from-image \
  --image <your-registry>/langgraph-cube:latest \
  --writable-layer-size 2G \
  --expose-port 49983 --probe 49983 --probe-path /health
```

Then set `CUBE_API_URL`, `CUBE_TEMPLATE_ID`, `CUBE_PROXY_NODE_IP`, and your LLM key. The variable
table in the LangChain guide applies unchanged.

### 3. Define the graph

The graph has three moving parts:

- **`coder`** — asks the LLM for a Python script and executes it in the sandbox.
- **`reviewer`** — asks the LLM whether the output answers the request.
- **a conditional edge** — routes `reviewer` back to `coder` (retry) or to `END`.

One sandbox is created for the whole run and reused across every `coder` invocation; its id is kept
in `State` so it survives across nodes and can be resumed later.

```python
from __future__ import annotations

import itertools
import os
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
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
    `sandbox_id` pins one MicroVM for the whole graph run so `pause()` /
    `connect()` can resume it later."""
    messages: Annotated[list, add_messages]
    sandbox_id: str
    attempts: int
    done: bool


def make_run_python(sandbox: Sandbox):
    """Return a `run_python` tool bound to one Cube sandbox — the same
    `cubesandbox` SDK pattern used by the LangChain guide."""
    _counter = itertools.count()

    def run_python(code: str) -> str:
        script = f"/workspace/_agent_{next(_counter)}.py"
        sandbox.files.write(script, code)
        result = sandbox.commands.run(f"python3 {script}", timeout=120, cwd="/workspace")
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
    "network access."
)

REVIEWER_PROMPT = (
    "You are a reviewer. Given the user request and the latest code output, decide "
    "whether the request is fully answered. Reply with exactly one word: DONE if the "
    "output answers the request, otherwise RETRY followed by one line on what is missing."
)


def coder(state: AgentState, run_python) -> dict:
    """Ask the LLM for code, execute it in the Cube sandbox, append the output."""
    code = llm.invoke(
        [{"role": "system", "content": CODER_PROMPT}, *state["messages"]]
    ).content
    output = run_python(code)
    return {"messages": [{"role": "assistant", "content": f"[code output]\n{output}"}]}


def reviewer(state: AgentState) -> dict:
    """Judge whether the latest output answers the request."""
    verdict = llm.invoke(
        [{"role": "system", "content": REVIEWER_PROMPT}, *state["messages"]]
    ).content.strip().upper()
    return {
        "messages": [{"role": "assistant", "content": f"[reviewer] {verdict}"}],
        "attempts": state.get("attempts", 0) + 1,
        "done": verdict.startswith("DONE"),
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


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else (
        "Load sales.csv from /workspace, compute total revenue per month, "
        "and report the month -> revenue numbers."
    )

    # One MicroVM for the whole run, reused across every coder -> reviewer loop.
    # The context manager tears it down on exit, so nothing is left behind.
    with Sandbox.create(template=os.environ["CUBE_TEMPLATE_ID"], timeout=600) as sandbox:
        run_python = make_run_python(sandbox)
        graph = build_graph(run_python)
        result = graph.invoke({
            "messages": [{"role": "user", "content": question}],
            "sandbox_id": sandbox.sandbox_id,
            "attempts": 0,
            "done": False,
        })
        for msg in reversed(result["messages"]):
            if msg.content:
                print(msg.content)
                break
        else:
            print("(no final answer)")
```

Run it:

```bash
pip install langgraph langchain-openai cubesandbox python-dotenv
python langgraph_agent_demo.py "Load sales.csv, compute total revenue per month."
```

### Expected behavior

`coder` writes the LLM-generated script to a fresh `/workspace/_agent_<n>.py` and runs it inside the
MicroVM; `reviewer` reads the output and either returns `DONE` or sends the graph back to `coder`
(up to 3 attempts). The `attempts` counter in `State` caps the loop so a stubborn request cannot spin
forever.

## Advanced: checkpointing + `pause()` / `connect()`

LangGraph checkpoints the graph state; Cube snapshots the sandbox. The two pair naturally for
long-running, resumable agents:

| LangGraph | Cube Sandbox |
|---|---|
| `builder.compile(checkpointer=MemorySaver())` | `Sandbox.create(template=...)` |
| `config = {"configurable": {"thread_id": "t1"}}` | `sandbox_id` in `State` |
| resume with `invoke(..., config)` | `sandbox.pause()` then `Sandbox.connect(sandbox_id)` |

```python
from langgraph.checkpoint.memory import MemorySaver

graph = build_graph(run_python, checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "t1"}}

with Sandbox.create(template=os.environ["CUBE_TEMPLATE_ID"], timeout=1800) as sandbox:
    graph.invoke(initial_input, config=config)      # checkpoint saved under thread_id t1
    sandbox.pause()
    sandbox = Sandbox.connect(sandbox.sandbox_id)   # /workspace intact after resume
    graph.invoke(follow_up_input, config=config)    # resumes the same thread_id
```

Keep the LangGraph `thread_id` aligned with the Cube `sandbox_id` (e.g. store both in your
orchestration layer) so a resumed graph reattaches to the same sandbox.

## Caveats

- **State is not serialized with the sandbox.** `State` lives in your process (or the LangGraph
  checkpointer); only `/workspace` persists across `pause()` / `connect()`. Persist anything the
  graph needs across a hard resume.
- **One sandbox, one graph run.** The `coder`/`reviewer` loop reuses a single MicroVM; do not create
  a new sandbox per node — you would pay the lifecycle cost every step.
- **State does not persist across tool calls.** Each `commands.run` is a fresh `python3` process;
  inline everything a snippet needs or write intermediate results back to `/workspace`.
- **Preinstall the stack into the image.** Under a default-deny egress policy a runtime `pip install`
  fails; bake pandas / numpy / matplotlib into the template.
- **Cap the retry loop.** Use the `attempts` counter (as above) or LangGraph's recursion limit, so a
  reviewer that keeps asking for retries cannot exhaust the sandbox `timeout`.

## References

- LangChain integration (the `create_agent` counterpart): [`langchain.md`](./langchain.md)
- Custom template images: [`docs/guide/tutorials/bring-your-own-image.md`](../tutorials/bring-your-own-image.md)
- Snapshot / clone / rollback: [`docs/guide/snapshot-rollback-clone.md`](../snapshot-rollback-clone.md)
- LangGraph: <https://github.com/langchain-ai/langgraph>
- E2B SDK: <https://github.com/e2b-dev/e2b>
