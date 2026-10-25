"""
LangGraph Agent 编排（v1.1 工具调用版）
--------------------------------------
用 LangChain 的 create_agent 构建带工具调用的 Agent：

    user question
        ↓
    ┌───────────── Agent (LLM) ─────────────┐
    │ 自己决定：                            │
    │  · 查知识库文档 → query_docs          │
    │  · 数学计算    → calculator           │
    │  · 都不用 → 直接回答                 │
    └─────────────┬─────────────────────────┘
                  ↓
             final answer

create_agent 内部是基于 LangGraph 的 ReAct 循环：
LLM 决定调哪个工具 → 执行工具 → 结果回填 → 再决定 → …直到给出最终答案。
"""
import re

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from . import config
from .rag import RAGEngine
from .tools import build_tools

# 工具返回文本里的来源标记：「（来源：a.txt, b.txt）」→ 提取出文件名
_SOURCES_RE = re.compile(r"来源：([^）)]+)")

SYSTEM_PROMPT = (
    "你是文档知识库问答助手，负责回答文档内容相关问题，也能做数学计算。\n"
    "工具选择规则：\n"
    "  - 用户问文档中的知识、概念、机制、参数等 → 调用 query_docs。\n"
    "  - 用户要求做数学计算（四则运算、幂、取模等）→ 调用 calculator。\n"
    "  - 其他闲聊可以直接回答，不需要调用工具。\n"
    "请用中文回答，简洁准确。引用文档内容时说明来源。"
)


def _build_llm() -> ChatOpenAI:
    """DeepSeek（OpenAI 兼容协议）聊天模型，支持工具调用"""
    if not config.DEEPSEEK_API_KEY:
        raise RuntimeError(
            "未设置 DEEPSEEK_API_KEY，请复制 .env.example 为 .env 并填入 Key"
        )
    return ChatOpenAI(
        model=config.DEEPSEEK_MODEL,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=0.3,
        timeout=config.LLM_TIMEOUT_S,
        max_retries=config.LLM_MAX_RETRIES,
    )


def build_agent(rag: RAGEngine):
    """构建带工具调用的 LangGraph Agent"""
    tools = build_tools(rag)
    return create_agent(
        model=_build_llm(),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        name="hybrid_rag_agent",
    )


def invoke_agent(agent, question: str) -> dict:
    """运行 agent，返回 {answer, tools_used, steps, sources}

    - answer:      最终回答
    - tools_used:  本次实际调用过的工具名（演示「Agent 自己决定用哪个」）
    - steps:       [{"tool", "args", "result"}, ...] 按调用顺序排列
    - sources:     本次回答引用的文档来源（从 query_docs 的返回文本中提取）
    """
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    messages = result["messages"]

    steps: list[dict] = []
    # 按 tool_call_id 配对：同名工具被调用多次时，按工具名做 key 会互相覆盖
    pending: dict[str, dict] = {}
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                step = {"tool": tc["name"], "args": tc.get("args", {}), "result": ""}
                steps.append(step)
                if tc.get("id"):
                    pending[tc["id"]] = step
        tool_call_id = getattr(msg, "tool_call_id", None)
        if tool_call_id and tool_call_id in pending:
            pending[tool_call_id]["result"] = str(msg.content)

    # 兜底：个别版本不返回 tool_call_id 时，退化为按工具名顺序回填
    if any(not s["result"] for s in steps):
        by_name: dict[str, list[str]] = {}
        for msg in messages:
            name = getattr(msg, "name", None)
            if name:
                by_name.setdefault(name, []).append(str(msg.content))
        used: dict[str, int] = {}
        for step in steps:
            if step["result"]:
                continue
            queue = by_name.get(step["tool"], [])
            i = used.get(step["tool"], 0)
            if i < len(queue):
                step["result"] = queue[i]
                used[step["tool"]] = i + 1

    sources: list[str] = []
    for step in steps:
        if step["tool"] != "query_docs":
            continue
        for matched in _SOURCES_RE.findall(step["result"]):
            for name in matched.split(","):
                name = name.strip()
                if name and name not in sources:
                    sources.append(name)

    tools_used = [s["tool"] for s in steps]

    final = messages[-1]
    answer = final.content if isinstance(final.content, str) else str(final.content)
    return {
        "answer": answer,
        "tools_used": tools_used,
        "steps": steps,
        "sources": sources,
    }
