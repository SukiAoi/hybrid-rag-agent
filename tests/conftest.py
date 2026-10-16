"""
pytest 全局配置
---------------
`app/main.py` 在**模块顶层**就执行了 `agent = build_agent(rag)`，而
`app/agent.py::_build_llm()` 在 `DEEPSEEK_API_KEY` 为空时会直接 `raise RuntimeError`。
也就是说：**环境里没有非空 Key 时，`import app.main` 就会失败**。

所以必须在导入任何 `app.*` 之前塞一个假 Key。
冒烟测试只打 `/health`，不会真的调用 DeepSeek，因此假 Key 完全够用。

⚠️ 用 setdefault 而不是直接赋值：本地/CI 若已配置真实 Key，不会被覆盖。
"""
import os
import sys
from pathlib import Path

# 允许从任意工作目录调用 pytest（不依赖 `python -m pytest` 注入 cwd）
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 必须在 import app.* 之前执行
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test-dummy")
