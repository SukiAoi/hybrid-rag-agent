"""
工具调用演示：Agent 自己决定用哪个工具 🛠️
=========================================

演示用例：
  1. 文档问题  → Agent 应该调用 query_docs（RAG）
  2. 数学问题  → Agent 应该调用 calculator
  3. 闲聊      → Agent 应该不调用任何工具

运行（在项目根目录）：
    & .venv\\Scripts\\python.exe demo_tools.py
"""
import logging

from app.agent import build_agent, invoke_agent
from app.rag import RAGEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CASES = [
    "攀爬系统有哪些状态？",
    "2 ** 10 + 5 * 3 等于多少？",
    "你好，今天天气怎么样？",
    "把跳跃高度从 3 改成 5 会影响什么？",
    "(1024 - 24) / 10 是多少？",
]


def fmt_steps(steps: list[dict]) -> str:
    lines = []
    for s in steps:
        args = "、".join(f"{k}={v}" for k, v in s.get("args", {}).items())
        result = str(s.get("result", ""))[:120]
        lines.append(f"      · 调用 {s['tool']}({args})\n        → {result}")
    return "\n".join(lines) if lines else "      （未调用工具，直接回答）"


def main() -> None:
    print("=" * 72)
    print("🛠️  OnlyUp! Agent 工具调用演示（v1.1）")
    print("=" * 72)

    rag = RAGEngine()
    rag.build()
    agent = build_agent(rag)
    print("✅ Agent 已就绪，工具：calculator, query_onlyup_docs\n")

    for q in CASES:
        print("-" * 72)
        print(f"❓ 问：{q}")
        result = invoke_agent(agent, q)
        print(f"🛠️  用：{result['tools_used'] or '（无）'}")
        if result["steps"]:
            print(fmt_steps(result["steps"]))
        print(f"💬 答：{result['answer']}")
        print()


if __name__ == "__main__":
    main()
