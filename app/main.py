"""
FastAPI 入口
------------
启动：  uvicorn app.main:app --reload
文档：   http://127.0.0.1:8000/docs
测试：   curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"出口欧盟需要哪些认证？\"}"
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import config
from .agent import build_agent, invoke_agent
from .landing import INDEX_HTML
from .rag import RAGEngine
from .ratelimit import RateLimiter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

rag = RAGEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时预热：构建/加载知识库（幂等），避免首请求卡顿"""
    logger.info("⏳ 预热：构建 / 加载知识库...")
    rag.build()
    logger.info("✅ 知识库就绪，agent 已编译")
    yield
    logger.info("👋 应用关闭")


app = FastAPI(title="Hybrid RAG Agent", version="1.1.0", lifespan=lifespan)
agent = build_agent(rag)
limiter = RateLimiter(
    per_min=config.RATE_LIMIT_PER_MIN,
    per_day=config.RATE_LIMIT_PER_DAY,
    global_day=config.GLOBAL_DAILY_QUOTA,
)


class AskRequest(BaseModel):
    # 边界校验：空问题会直接调用一次 LLM，超长问题会烧掉大量 token
    question: str = Field(
        ...,
        min_length=1,
        max_length=config.MAX_QUESTION_LEN,
        description=f"用户问题，1–{config.MAX_QUESTION_LEN} 字",
    )


class ToolStep(BaseModel):
    tool: str
    args: dict
    result: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str] = []
    tools_used: list[str] = []
    steps: list[ToolStep] = []


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    """首页：给非技术访客一个能直接提问的界面（默认的 404 JSON 不适合演示）"""
    return INDEX_HTML


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "kb_chunks": len(rag.all_documents)}


@app.get("/stats", include_in_schema=False)
def stats() -> dict:
    """当日配额使用情况（公开只读，不含敏感信息）"""
    return limiter.snapshot()


def _client_ip(request: Request) -> str:
    """取真实客户端 IP。

    经过 Caddy 反代后 request.client.host 是容器网关 IP，必须读转发头。
    优先用 Caddy 写入的 X-Real-IP（用 header_up 覆盖，客户端伪造不了）；
    X-Forwarded-For 由客户端可控，只在没有 X-Real-IP 时兜底。
    """
    real = request.headers.get("x-real-ip")
    if real:
        return real.strip()
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request) -> AskResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="问题不能为空")

    ip = _client_ip(request)
    allowed, reason = limiter.check(ip)
    if not allowed:
        logger.warning("⛔ 限流命中 ip=%s reason=%s", ip, reason)
        raise HTTPException(status_code=429, detail=reason)

    logger.info("❓ 收到问题：%s", question)
    result = invoke_agent(agent, question)
    tools_used = result["tools_used"]
    logger.info("🛠️ 使用的工具：%s", tools_used or "（无，直接回答）")
    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        tools_used=tools_used,
        steps=[ToolStep(**s) for s in result["steps"]],
    )
