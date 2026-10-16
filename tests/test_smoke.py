"""
冒烟测试 / Smoke test
--------------------
只验证「服务能起来 + 知识库加载成功」，即容器化部署后最基本的可用性。

🔴 不要在这里测 `/ask`：那条路径会真实调用 DeepSeek API，
   既产生费用又需要真实 Key，不适合放在自动化测试里。

前置依赖（见 tests/conftest.py）：导入 app 前必须先注入假 DEEPSEEK_API_KEY，
否则 `app/main.py` 模块顶层的 build_agent() 会直接抛 RuntimeError。
"""
from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint():
    """GET /health 应返回 200，且知识库已加载出非空文档块"""
    # with 块会触发 lifespan，即完成知识库预热；健康探针必须在预热之后才有意义
    with TestClient(app) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["kb_chunks"] > 0
