"""
Agent 工具集
------------
给 Agent 提供的两个工具，让它能自己决定用哪个：

  1. query_docs —— 查知识库文档（RAG）
  2. calculator —— 安全计算数学表达式

设计要点：
  - 用 langchain_core.tools.tool 装饰器自动生成输入 schema
  - 计算器用 ast 安全求值，避免 eval 注入
  - 文档工具通过闭包拿到 RAGEngine，惰性构建知识库
"""
import ast
import logging
import operator as op
import re

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# 计算器允许的运算（仅数字/运算符，杜绝任意代码执行）
_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.BitAnd: op.and_,
    ast.BitOr: op.or_,
    ast.BitXor: op.xor,
    ast.LShift: op.lshift,
    ast.RShift: op.rshift,
}
_ALLOWED_UNARYOPS = {ast.USub: op.neg, ast.UAdd: op.pos}

# 只允许数字 + 运算符字符，先过滤掉字母/下划线等
_EXPR_ALLOWED = re.compile(r"^[0-9+\-*/().%<>~&|^ \t]+$")


# ---- 规模上限：白名单只保证「不执行任意代码」，但挡不住「合法字符算出天文数字」 ----
# 例如 9**9**9**9 或 1<<100000000 会让进程算到卡死 / OOM，必须在求值前拦掉。
MAX_EXPR_LEN = 200          # 表达式最大长度
MAX_POW_EXPONENT = 1000     # 指数上限（2**1000 以内，足够日常使用）
MAX_SHIFT_BITS = 64         # 位移上限
MAX_RESULT_BITS = 4096      # 中间结果最大位数（约 1233 位十进制）


def _check_magnitude(value):
    """拦截过大的整数中间结果，避免后续运算指数级膨胀"""
    if isinstance(value, int) and abs(value).bit_length() > MAX_RESULT_BITS:
        raise ValueError("结果过大，已超出计算器允许的范围")
    return value


def _safe_eval(node):
    """递归求值 AST，只放行白名单内的节点"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        # 幂与位移要先检查操作数规模，不能先算再拦（算出结果时已经晚了）
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_POW_EXPONENT:
                raise ValueError(f"指数过大（上限 {MAX_POW_EXPONENT}）")
        elif isinstance(node.op, (ast.LShift, ast.RShift)):
            if abs(right) > MAX_SHIFT_BITS:
                raise ValueError(f"位移过大（上限 {MAX_SHIFT_BITS} 位）")
        return _check_magnitude(_ALLOWED_BINOPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"不支持的表达式节点：{type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """计算数学表达式（支持四则运算、幂、取模、位运算、括号）。

    示例："2 ** 10"、"((1 + 2) * 3) / 4"、"17 % 5"。
    只接受数字和运算符，其余字符会报错。
    """
    expr = expression.strip().replace("×", "*").replace("÷", "/").replace("^", "**")
    if len(expr) > MAX_EXPR_LEN:
        return f"错误：表达式过长（上限 {MAX_EXPR_LEN} 个字符）。"
    if not _EXPR_ALLOWED.fullmatch(expr):
        return "错误：表达式包含不允许的字符，请只输入数字和运算符。"
    try:
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval(tree)
        return f"{expr} = {result}"
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, RecursionError) as exc:
        return f"错误：无法计算（{exc}）"


def build_tools(rag):
    """根据 RAGEngine 构造完整工具列表（文档工具依赖 rag 实例）"""

    @tool
    def query_docs(question: str) -> str:
        """查询知识库文档。

        任何与知识库文档内容相关的问题都应使用本工具。
        返回基于文档片段的回答，并附上来源文件。
        """
        if rag is None:
            return "知识库尚未初始化。"
        contexts = rag.retrieve(question)
        if not contexts:
            return "文档中未找到相关信息。"
        answer, sources = rag.answer_with_contexts(question, contexts)
        source_line = f"（来源：{', '.join(sources)}）" if sources else ""
        return f"{source_line}\n{answer}"

    return [calculator, query_docs]
