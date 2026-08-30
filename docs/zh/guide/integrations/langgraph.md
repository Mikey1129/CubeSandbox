---
title: LangGraph 集成指南
author: Mikey1129
date: 2026-08-29
tags:
  - integration
  - langgraph
  - agent
lang: zh-CN
---

# LangGraph 集成指南

将一个 [LangGraph](https://github.com/langchain-ai/langgraph) Agent——由节点与条件边构成的显式图——运行在
[CubeSandbox](https://github.com/TencentCloud/CubeSandbox) 的 MicroVM 中，让它在沙箱内执行 Python。由于
Cube 暴露了**与 E2B 兼容的 API**，代码执行工具可以从 E2B 无缝切换到 Cube，同时为 Agent 生成的每一行
代码获得 KVM 级隔离。

本文是 [LangChain 集成](./langchain.md)的 LangGraph 对应版本。LangChain 指南使用高层封装
`create_agent`，本文则用 `StateGraph` **显式**搭建图：由你掌控控制流、在节点间共享同一个沙箱，并把
LangGraph 的 checkpoint 机制与 Cube 的 `pause()` / `connect()` 对接。

本指南的可运行版本见
[`examples/langgraph-integration`](https://github.com/TencentCloud/CubeSandbox/tree/master/examples/langgraph-integration)，
它是下方代码清单的权威来源。

## LangGraph 与 `create_agent` 的对比

| | `create_agent`（LangChain 指南） | 显式 `StateGraph`（本文） |
|---|---|---|
| 图结构 | 固定的工具调用循环，对你是黑盒 | 每个节点、每条边都由你定义 |
| 重试 / 循环 | 隐含在 Agent 循环内部 | 显式的 `add_conditional_edges` |
| 状态 | 不透明的消息历史 | 你设计的类型化 `State`（重试次数、判定结果…） |
| 恢复 | 不直接暴露 | `checkpointer` 对接 Cube 的 `pause()` / `connect()` |

当你只需要一个能调用工具的 Agent 时，用 `create_agent` 即可。当你想构建「生成 → 执行 → 审查 → 重试」
这样的多步工作流、让沙箱跨阶段复用、并在之后可通过 checkpoint 恢复运行时，请使用显式的 `StateGraph`。

## 集成对象与版本

| 组件 | 版本 | 说明 |
|---|---|---|
| langgraph | `>=0.2` | `StateGraph`、`START`/`END`、`add_messages` |
| langchain-openai | `>=1.0,<2.0` | `ChatOpenAI`（任意 OpenAI 兼容端点） |
| cubesandbox SDK | `>=0.6.0` | `Sandbox.create` / `files.write` / `commands.run` |
| CubeSandbox 平台 | `>=0.3.0` | 核心；可选特性见 LangChain 指南 |
| CubeSandbox 基础镜像 | `ghcr.io/tencentcloud/cubesandbox-base:2026.16` | 在其上叠加 Python 栈 |

## 前置条件

前置条件与 LangChain 指南完全一致——同一个沙箱模板对两者都适用：

- 已部署 CubeSandbox，CubeAPI 可从 `http://<node>:3000` 访问。
- 已构建并注册一个含 Python 栈（pandas / numpy / matplotlib / scikit-learn）的模板镜像。可参考
  LangChain 指南的[模板镜像](../integrations/langchain.md)步骤，或直接复用同一个 template id。
- `cubesandbox` SDK 所需环境变量：`CUBE_API_URL`、`CUBE_TEMPLATE_ID`、`CUBE_PROXY_NODE_IP`；
  CubeAPI 后端启用鉴权时还需 `CUBE_API_KEY`（未设置时 SDK 不发送鉴权头）。
- Python 3.10+（示例使用 `str | None`、`Annotated` 及 `langchain-openai` 1.x）。
- 经 `OPENAI_BASE_URL` / `OPENAI_API_KEY`（或 `TOKENHUB_API_KEY`）接入的 OpenAI 兼容 LLM 端点。

## 接入步骤

### 1. 构建模板镜像

复用 LangChain 指南的 `Dockerfile`（在 `cubesandbox-base` 之上叠加 Python 数据科学栈，envd 监听
`:49983`）。镜像里无需烘焙任何 LangGraph 专属依赖——图在宿主机上运行，只有**代码执行**发生在沙箱内。
用第 2 步要注册的 tag 构建并推送：

```bash
docker build --platform linux/amd64 -t <your-registry>/langgraph-cube:latest <path-to-dockerfile>
docker push <your-registry>/langgraph-cube:latest
```

### 2. 注册模板并配置环境变量

```bash
cubemastercli tpl create-from-image \
  --image <your-registry>/langgraph-cube:latest \
  --writable-layer-size 2G \
  --expose-port 49983 --probe 49983 --probe-path /health
```

随后设置 `CUBE_API_URL`、`CUBE_TEMPLATE_ID`、`CUBE_PROXY_NODE_IP` 以及 LLM key。LangChain 指南中的
环境变量表此处完全适用。

### 3. 定义图

图由三个活动部分组成：

- **`coder`** —— 让 LLM 生成 Python 脚本，并在沙箱内执行。
- **`reviewer`** —— 让 LLM 判断输出是否回答了请求。
- **一条条件边** —— 将 `reviewer` 路由回 `coder`（重试）或路由到 `END`。

整个运行只创建一个沙箱，并在每次 `coder` 调用间复用；图通过 `run_python` 闭包持有它（而非通过 `State`），它的 id 正是之后通过 checkpoint 恢复运行所需的键。

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

for v in ("CUBE_TEMPLATE_ID", "CUBE_PROXY_NODE_IP"):
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
    """跨节点共享的状态。`messages` 累积对话；`attempts` / `done` 驱动重试循环。
    沙箱本身通过 `run_python` 闭包共享，而非通过 `State`。"""
    messages: Annotated[list, add_messages]
    attempts: int
    done: bool


def make_run_python(sandbox: Sandbox):
    """返回一个绑定到单个 Cube 沙箱的 `run_python` 工具——与 LangChain 指南使用
    相同的 `cubesandbox` SDK 模式。"""
    _counter = itertools.count()

    def run_python(code: str) -> str:
        script = f"/workspace/_agent_{next(_counter)}.py"
        try:
            sandbox.files.write(script, code)
            result = sandbox.commands.run(f"python3 {script}", timeout=120, cwd="/workspace")
        except Exception as exc:
            # 沙箱已死、写入失败、传输错误和单命令超时都会在这里抛异常；把它作为
            # 工具输出返回，让 reviewer 能看到失败并决定重试，而不是让异常直接终止整张图的运行。
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
    """从消息内容中提取纯文本；content 可能是 str、单个 content block dict，或 block 列表。"""
    if not content:
        return ""                     # None / 空内容——视为无文本，避免返回字面量 "None"
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]           # 单个 content block，而非列表
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def strip_code_fence(text: str) -> str | None:
    """返回第一个 Markdown 围栏内的代码；若没有围栏则返回 None。
    模型常在围栏块前加上说明文字，因此不能假设回复以围栏开头。"""
    fence = "`" * 3                      # 三个反引号，避免字面量围栏
    text = text.strip()
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(fence):
            start = i
            break
    if start is None:
        return None
    inner = []
    for line in lines[start + 1:]:
        # 关闭条件：以三个及以上反引号开头，且该行其余部分为空（裸 ``` 行），
        # 或同行的只是文字（模型偶尔会滑成 `` ``` Done! ``）。脚本内部像
        # ```markdown 这样的语言标签行要当作代码保留。（已知限制：docstring 里的
        # 裸 ``` 行仍会被提前当作关闭。）
        s = line.strip()
        if s.startswith(fence) and set(s.split(None, 1)[0]) == {"`"}:
            break
        inner.append(line)
    return "\n".join(inner).strip()


def coder(state: AgentState, run_python) -> dict:
    """让 LLM 生成代码，在 Cube 沙箱内执行，并追加输出。"""
    code = strip_code_fence(extract_text(llm.invoke(
        [{"role": "system", "content": CODER_PROMPT}, *state["messages"]]
    ).content))
    if code is None:
        # 模型没有返回围栏代码块；直接提示，而不是把散文写进 .py 文件、浪费一次重试机会。
        return {"messages": [{"role": "assistant",
                              "content": "[code output]\n(no code block in model reply)"}]}
    output = run_python(code)
    # 同时截断代码与输出，以免超大结果（如打印整个 DataFrame）或过长的生成脚本
    # 在多次重试中撑爆模型上下文窗口；各自保留尾部——打印的数字（以及
    # stderr/traceback）在末尾，脚本尾部仍能展示实际运行的逻辑。
    if len(code) > 4000:
        code = "[earlier code truncated]\n" + code[-4000:]
    if len(output) > 4000:
        output = "[earlier output truncated]\n" + output[-4000:]
    # 把代码连同输出一起写入消息，这样 RETRY 时 coder 能看到上次写了什么，
    # 而不是重复犯同样的错误。
    return {"messages": [{"role": "assistant",
                          "content": f"[code]\n{code}\n[code output]\n{output}"}]}


def reviewer(state: AgentState) -> dict:
    """判断最新输出是否回答了请求。"""
    verdict = extract_text(llm.invoke(
        [{"role": "system", "content": REVIEWER_PROMPT}, *state["messages"]]
    ).content).strip().upper()
    # prompt 要求回复以一个判定词（DONE/RETRY）开头，所以只按第一个 token 判定，
    # 并剥离 markdown 装饰；不要扫描整句，否则 RETRY 的解释文字里出现 "done" 会误判。
    first = (verdict.split() or [""])[0].strip(":*#`")
    done = first.rstrip(".,!?;") == "DONE"
    # 把判定作为 user 角色消息发出，让 coder 把 RETRY 当作需要修复的指令，
    # 而不是当作它自己先前的 assistant 输出。
    return {
        "messages": [{"role": "user", "content": f"[reviewer] {verdict}"}],
        "attempts": state.get("attempts", 0) + 1,
        "done": done,
    }


def route_after_review(state: AgentState) -> str:
    """条件边：重试 `coder`，或达到次数上限后结束。"""
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

    # 整个运行只用一个 MicroVM，在 coder -> reviewer 循环中反复复用。
    # 上下文管理器在退出时销毁沙箱，不会留下残留。
    with Sandbox.create(template=os.environ["CUBE_TEMPLATE_ID"], timeout=600) as sandbox:
        run_python = make_run_python(sandbox)
        graph = build_graph(run_python)
        result = graph.invoke({
            "messages": [{"role": "user", "content": question}],
            "attempts": 0,
            "done": False,
        })
        # 最后一条消息是 reviewer 的判定；改为打印代码输出，让用户真正看到计算结果。
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
```

将上面的代码保存为 `langgraph_agent_demo.py`，然后运行：

```bash
pip install "langgraph>=0.2,<2" "langchain-openai>=1.0,<2.0" "cubesandbox>=0.6.0" python-dotenv
python langgraph_agent_demo.py "Load sales.csv, compute total revenue per month."
```

### 预期行为

`coder` 将 LLM 生成的脚本写入全新的 `/workspace/_agent_<n>.py` 并在 MicroVM 内运行；`reviewer` 读取
输出，要么返回 `DONE`，要么把图送回 `coder`（最多 3 次）。`State` 中的 `attempts` 计数器给循环设了上限，
避免一个执拗的请求无限循环下去。

## 进阶：checkpoint 与 `pause()` / `connect()`

LangGraph 对图状态做 checkpoint，Cube 对沙箱做快照，二者天然互补，适合长时间、可恢复的 Agent：

> 下面的 `MemorySaver` 把 checkpoint 存在**进程内存**里——适合 demo，但进程重启后即丢失。
> 如需跨进程恢复，请改用持久化 checkpointer，例如 `langgraph-checkpoint-sqlite` /
> `-postgres` 提供的 `SqliteSaver` / `PostgresSaver`。

| LangGraph | Cube Sandbox |
|---|---|
| `builder.compile(checkpointer=MemorySaver())` | `Sandbox.create(template=...)` |
| `config = {"configurable": {"thread_id": sandbox.sandbox_id}}` | `sandbox.sandbox_id` |
| 在同一 thread 上用 `invoke(..., config)` 开始新一轮 | `sandbox.pause()` 后 `Sandbox.connect(sandbox_id)` |

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()  # 进程内 demo；真正的恢复需改用持久化 checkpointer


def stage_input(messages):
    """构建单阶段的图输入。`attempts`/`done` 没有 reducer，显式传值会覆盖 checkpoint 里的旧值；
    `messages` 使用 `add_messages`，新消息会追加到之前的历史——若想从干净状态开始，请换新的 thread_id。"""
    return {"messages": messages, "attempts": 0, "done": False}


sandbox = Sandbox.create(template=os.environ["CUBE_TEMPLATE_ID"], timeout=1800)
# 以 sandbox_id 作为 checkpoint 线程的键，使同一个 thread_id 在 pause() / connect()
# 之后重新挂载到同一个 MicroVM。
config = {"configurable": {"thread_id": sandbox.sandbox_id}}
try:
    graph = build_graph(make_run_python(sandbox), checkpointer=checkpointer)
    graph.invoke(stage_input([{"role": "user", "content": "first task"}]), config=config)

    sandbox.pause()                                   # 快照 VM + 根文件系统
    # Sandbox.connect() 返回的是新实例；run_python 闭包捕获的是暂停前的旧实例，
    # 因此需要重新绑定工具并在新实例上重建图，同时保留同一个 checkpointer 以恢复同一 checkpoint 线程。
    sandbox = Sandbox.connect(sandbox.sandbox_id)     # 恢复后 /workspace 保持不变
    graph = build_graph(make_run_python(sandbox), checkpointer=checkpointer)
    graph.invoke(stage_input([{"role": "user", "content": "follow-up task"}]), config=config)
finally:
    sandbox.kill()
```

让 LangGraph 的 `thread_id` 与 Cube 的 `sandbox_id` 保持一致（例如都存放在你的编排层），这样恢复的图才能
重新挂载到同一个沙箱上。由于 `MemorySaver` 只在当前进程内有效，跨重启恢复还需把编排层与持久化
checkpointer（`SqliteSaver` / `PostgresSaver`）配合使用。

## 注意事项

- **State 不会随沙箱序列化。** `State` 存在于你的进程（或 LangGraph 的 checkpointer）中；只有 `/workspace`
  会在 `pause()` / `connect()` 之间保留。任何需要在硬恢复后仍存在的图状态，都要自行持久化。
- **一个沙箱、一次图运行。** `coder`/`reviewer` 循环复用同一个 MicroVM；不要每个节点新建沙箱——那样每一步
  都要付一次生命周期开销。
- **工具调用之间状态不保留。** 每次 `commands.run` 都是全新的 `python3` 进程；代码片段所需内容都要内联，
  或把中间结果写回 `/workspace`。
- **把栈预先装进镜像。** 在默认拒绝出口流量的策略下，运行时的 `pip install` 会失败；请把 pandas / numpy /
  matplotlib 烘焙进模板。
- **给重试循环设上限。** 用 `attempts` 计数器（如上）或 LangGraph 的递归限制，避免一直要求重试的 reviewer
  耗尽沙箱的 `timeout`。
- **`MemorySaver` 仅限进程内。** checkpoint 存在内存里，进程重启即丢失；即使 Cube 沙箱本身能在
  `pause()` / `connect()` 后存续，跨进程恢复也需改用 `SqliteSaver` / `PostgresSaver`（来自
  `langgraph-checkpoint-sqlite` / `-postgres`）。

## 参考资料

- LangChain 集成（`create_agent` 对应版本）：[`langchain.md`](./langchain.md)
- 自定义模板镜像：[`docs/guide/tutorials/bring-your-own-image.md`](../tutorials/bring-your-own-image.md)
- 快照 / 克隆 / 回滚：[`docs/guide/snapshot-rollback-clone.md`](../snapshot-rollback-clone.md)
- LangGraph：<https://github.com/langchain-ai/langgraph>
- E2B SDK：<https://github.com/e2b-dev/e2b>
