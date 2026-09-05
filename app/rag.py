""" 
RAG 检索模块
------------
「混合检索 + RRF 融合 + LLM ReRank」思路，面向本地文档语料（data/*.txt）。

架构：
    问题 → BM25关键词召回 + 语义向量召回 → RRF 融合去重 → LLM ReRank → Top-K

踩坑提醒（详见 PITFALLS.md）：
  1. DeepSeek 没有 Embedding API → 用 ChromaDB 内置 ONNXMiniLM_L6_V2（本地 ONNX）
  2. 模型懒加载：避免 import 时卡在模型下载
  3. HuggingFace 国内下载慢 → 优先 ONNX（走 ChromaDB CDN，~23MB）
"""
import logging

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from rank_bm25 import BM25Okapi

from . import config

logger = logging.getLogger(__name__)

COLLECTION_NAME = "knowledge_docs"


class RAGEngine:
    """混合检索引擎：ChromaDB 语义 + BM25 关键词 + RRF 融合 + LLM ReRank"""

    def __init__(self, data_dir=None, chroma_dir=None, embedding_model=None):
        self.data_dir = data_dir or config.DATA_DIR
        self.chroma_dir = chroma_dir or config.CHROMA_DIR
        # 可选：'onnx'（默认英文）| 'bge'（中文 BAAI/bge-small-zh-v1.5）
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        # 懒加载（首次使用时初始化）
        self.embedding_fn = None
        self.llm = None
        self.collection = None
        self.bm25 = None
        self.all_documents: list[str] = []   # 所有块文本（供 BM25 分词）
        self.doc_sources: dict[int, str] = {}  # 块索引 → 来源文件名
        self._ready = False

    # ================= 模型懒加载 =================
    def _ensure_models(self) -> None:
        if self._ready:
            return
        # Embedding：默认本地 ONNX（英文），可切换 bge 中文模型（需 torch）
        if self.embedding_model == "bge":
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )

            self.embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name="BAAI/bge-small-zh-v1.5",
                normalize_embeddings=True,
            )
            logger.info("✅ 嵌入模型：BAAI/bge-small-zh-v1.5（中文，sentence-transformers）")
        else:
            try:
                self.embedding_fn = embedding_functions.ONNXMiniLM_L6_V2()
                logger.info("✅ 嵌入模型：ONNXMiniLM_L6_V2（本地 ONNX）")
            except Exception as exc:  # noqa: BLE001
                logger.warning("⚠️ ONNX 不可用（%s），回退默认嵌入模型", exc)
                self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        # LLM：DeepSeek（OpenAI 兼容协议），用于生成与 ReRank
        if not config.DEEPSEEK_API_KEY:
            raise RuntimeError("未设置 DEEPSEEK_API_KEY，请复制 .env.example 为 .env 并填入 Key")
        self.llm = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
        self._ready = True

    # ================= 1. 文档切块 =================
    @staticmethod
    def _load_and_split(file_path, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        chunks, start = [], 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({"text": chunk_text, "source": file_path.name})
            start += chunk_size - overlap
        return chunks

    # ================= 2. 构建 / 加载知识库 =================
    def build(self) -> None:
        """幂等构建：已有集合则加载，否则从 data/*.txt 构建"""
        self._ensure_models()
        logger.info("📚 构建知识库...")
        chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))

        try:
            self.collection = chroma_client.get_collection(
                name=COLLECTION_NAME, embedding_function=self.embedding_fn
            )
            existing = self.collection.get()
            self.all_documents = existing["documents"] or []
            metas = existing["metadatas"] or []
            self.doc_sources = {
                i: (m.get("source") or "unknown") for i, m in enumerate(metas)
            }
            logger.info("   ✅ 已有知识库，加载 %d 个文档块", len(self.all_documents))
        except chromadb.errors.NotFoundError:
            self.collection = chroma_client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_fn,
                metadata={"description": "文档知识库"},
            )
            all_chunks = []
            for txt_file in sorted(self.data_dir.glob("*.txt")):
                chunks = self._load_and_split(txt_file)
                logger.info("   📄 %s: %d 个块", txt_file.name, len(chunks))
                all_chunks.extend(chunks)
            for i, chunk in enumerate(all_chunks):
                self.collection.add(
                    ids=[f"chunk_{i}"],
                    documents=[chunk["text"]],
                    metadatas=[{"source": chunk["source"]}],
                )
            self.all_documents = [c["text"] for c in all_chunks]
            self.doc_sources = {i: c["source"] for i, c in enumerate(all_chunks)}
            logger.info("   ✅ 已存入 %d 个文档块", len(self.all_documents))

        # BM25 关键词索引（中文块先按字符拆分，兼顾关键词匹配）
        tokenized = [list(doc) for doc in self.all_documents]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("   ✅ BM25 索引就绪（%d 篇）", len(self.all_documents))

    # ================= 3. 双路召回 =================
    def _bm25_search(self, query: str, top_k: int = config.BM25_TOP_K):
        tokenized_query = list(query)
        scores = self.bm25.get_scores(tokenized_query)
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(i, scores[i]) for i in top if scores[i] > 0]

    def _semantic_search(self, query: str, top_k: int = config.SEMANTIC_TOP_K):
        results = self.collection.query(query_texts=[query], n_results=top_k)
        out = []
        for i in range(len(results["documents"][0])):
            text = results["documents"][0][i]
            dist = results["distances"][0][i] if results.get("distances") else 0
            out.append((text, dist))
        return out

    # ================= 4. RRF 融合去重 =================
    def _reciprocal_rank_fusion(self, bm25_results, semantic_results, k=config.RRF_K):
        """score(d) = Σ 1 / (k + rank_i(d))"""
        rrf_scores: dict[int, float] = {}
        doc_info: dict[int, str] = {}
        for rank, (idx, _) in enumerate(bm25_results, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank)
            if idx >= 0:
                doc_info[idx] = self.all_documents[idx]
        for rank, (text, _dist) in enumerate(semantic_results, start=1):
            try:
                idx = self.all_documents.index(text)
            except ValueError:
                continue
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank)
            doc_info[idx] = text

        merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {
                "index": i,
                "text": doc_info[i],
                "source": self.doc_sources.get(i, "unknown"),
                "rrf_score": s,
            }
            for i, s in merged
            if i < len(self.all_documents)
        ]

    # ================= 5. 检索（融合 + ReRank） =================
    def retrieve(self, query: str, top_k: int = config.FINAL_TOP_K) -> list[dict]:
        """返回 ReRank 后的 Top-K 候选块，按相关度降序"""
        self._ensure_models()
        if self.bm25 is None or self.collection is None:
            self.build()

        bm25_results = self._bm25_search(query)
        semantic_results = self._semantic_search(query)
        fused = self._reciprocal_rank_fusion(bm25_results, semantic_results)

        if len(fused) <= top_k:
            return fused
        # LLM ReRank：让模型给候选块按「与问题的相关度」打分排序
        ranked = self._llm_rerank(query, fused, top_k)
        return ranked

    def _llm_rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        lines = "\n".join(
            f"[{i}] {c['text'][:200]}..." for i, c in enumerate(candidates)
        )
        prompt = (
            "你是文档检索排序器。给定问题与候选文档片段，请只输出最相关的 "
            f"{top_k} 个片段编号（用逗号分隔，按相关度从高到低）。\n\n"
            f"问题：{query}\n\n候选片段：\n{lines}\n\n最相关的编号："
        )
        try:
            resp = self.llm.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=50,
            )
            picks = []
            for tok in resp.choices[0].message.content.split(","):
                tok = tok.strip()
                if tok.isdigit() and 0 <= int(tok) < len(candidates):
                    picks.append(candidates[int(tok)])
            if picks:
                return picks
        except Exception as exc:  # noqa: BLE001
            logger.warning("ReRank 失败（%s），退回 RRF 结果", exc)
        return candidates[:top_k]

    # ================= 6. 生成回答 =================
    def answer_with_contexts(self, query: str, contexts: list[dict]) -> tuple[str, list[str]]:
        """基于候选块生成回答，返回 (answer, sources)"""
        self._ensure_models()
        context_text = "\n\n---\n\n".join(c["text"] for c in contexts)
        sources = sorted({c["source"] for c in contexts if c.get("source")})
        prompt = (
            "你是一个 OnlyUp! 游戏设计文档助手。请严格依据下面的文档片段回答用户问题。\n"
            "如果片段中没有相关信息，就明确说“文档中未找到相关信息”，不要编造。\n\n"
            f"文档片段：\n{context_text}\n\n"
            f"用户问题：{query}\n\n回答："
        )
        resp = self.llm.chat.completions.create(
            model=config.DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip(), sources
