"""全局配置：路径 / API Key / 检索参数"""
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------- 路径 ----------
# 项目根目录（hybrid-rag-agent/）
BASE_DIR = Path(__file__).resolve().parent.parent

# 显式加载项目根目录的 .env：不依赖启动时的工作目录，从任意路径启动都能读到 Key
load_dotenv(BASE_DIR / ".env")

# 原始文档目录（放 *.txt 文档）
DATA_DIR = BASE_DIR / "data"
# ChromaDB 持久化目录（运行时生成）
CHROMA_DIR = BASE_DIR / "chroma_db"
# 中文 embedding 模型专用库（与英文 ONNX 库隔离，维度不同）
CHROMA_DIR_BGE = BASE_DIR / "chroma_db_bge"

# ---------- Embedding 模型 ----------
# 可选值：onnx（默认，ONNXMiniLM_L6_V2 英文，无需 torch）
#         bge（BAAI/bge-small-zh-v1.5 中文，需 sentence-transformers + torch）
# 生产环境默认用 onnx；要跑中文语义检索对比时设 EMBEDDING_MODEL=bge
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "onnx")

# ---------- LLM（DeepSeek，OpenAI 兼容协议） ----------
# 注意：DeepSeek 目前没有 Embedding API（见 PITFALLS.md），
#       向量化用本地 ONNX 模型，只有生成/ReRank 走 DeepSeek。
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# 超时与重试：SDK 默认超时可达 600s，上游卡住会占满线程池，必须显式收窄
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "30"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

# 单次提问的字符上限（防止超长输入把 token 账单打爆）
MAX_QUESTION_LEN = int(os.getenv("MAX_QUESTION_LEN", "500"))

# ---------- 检索参数 ----------
CHUNK_SIZE = 300        # OnlyUp 文档较长，块稍大
CHUNK_OVERLAP = 80      # 块重叠
BM25_TOP_K = 5          # BM25 关键词召回
SEMANTIC_TOP_K = 5      # 语义向量召回
FINAL_TOP_K = 3         # ReRank 后最终返回给 LLM 的块数
RRF_K = 60              # RRF 平滑参数（score = Σ 1/(k+rank)）
# ---------- 限流与成本保护 ----------
# 单 IP 每分钟最多几次（防止脚本连发）
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "3"))
# 单 IP 每天最多几次
RATE_LIMIT_PER_DAY = int(os.getenv("RATE_LIMIT_PER_DAY", "20"))
# 全站每天总次数上限（最硬的一道闸，超了就停，保护余额）
GLOBAL_DAILY_QUOTA = int(os.getenv("GLOBAL_DAILY_QUOTA", "300"))
# 单次回答的最大输出 token（限制成本，也避免长文拖慢响应）
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "800"))
# ---------- 知识库：额外把项目自身文档也入库 ----------
# 目的：让线上 demo 能回答"这个项目怎么部署、为什么这么设计、踩过什么坑"，
# 相当于一个可交互的项目介绍。
# 只保留 README（覆盖技术栈、评测、部署概览）；
# PITFALLS / deploy / demo_qa 会让业务类问题的召回变噪，故不入选。
# 要去掉项目自述，把 EXTRA_DOCS 设为 [] 即可。
EXTRA_DOCS = [
    "README.md",
]

# 语料版本号：内容变更时把它 +1，会自动建一个新集合，避免读到旧索引。
# （Chroma 的集合名固定，不换名就会一直加载旧的 13 个块）
CORPUS_VERSION = os.getenv("CORPUS_VERSION", "5")