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
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from . import config
from .rag import RAGEngine
from .tools import build_tools

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
    """运行 agent，返回 {answer, tools_used, steps}

    - answer:      最终回答
    - tools_used:  本次实际调用过的工具名（演示「Agent 自己决定用哪个」）
    - steps:       [{"tool", "args", "result"}, ...] 按调用顺序排列
    """
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    messages = result["messages"]

    steps: list[dict] = []
    tool_returns: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                steps.append(
                    {"tool": tc["name"], "args": tc.get("args", {}), "result": ""}
                )
        if getattr(msg, "name", None) is not None:
            tool_returns[msg.name] = str(msg.content)

    for step in steps:
        step["result"] = tool_returns.get(step["tool"], "")

    tools_used = [s["tool"] for s in steps]

    final = messages[-1]
    answer = final.content if isinstance(final.content, str) else str(final.content)
    return {"answer": answer, "tools_used": tools_used, "steps": steps}
