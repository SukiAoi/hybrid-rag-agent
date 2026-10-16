# Hybrid RAG Agent —— 容器化镜像
# 构建： docker build -t hybrid-rag-agent .
# 运行： docker run --env-file .env -p 8000:8000 hybrid-rag-agent
FROM python:3.11-slim

# 不写 .pyc、日志不缓冲（容器里日志能实时刷出）、pip 不留缓存
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖，再拷源码：源码改动不会让依赖层失效，构建更快
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 只拷贝运行必需的资源
COPY app/ ./app/
COPY data/ ./data/
COPY .env.example ./

EXPOSE 8000

# 镜像里没有 curl，用标准库做探针
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
