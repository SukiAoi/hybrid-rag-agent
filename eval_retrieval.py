"""
检索命中率评估脚本（Retrieval Hit Rate Evaluation）
====================================================
对 hybrid-rag-agent 的混合检索（BM25 + 语义向量 + RRF 融合）做量化评估，
统计 Hit Rate@1 / @3 / @5 与 MRR，并把结果写入 docs/RETRIEVAL_EVAL.md。

指标定义（简历可直接引用的检索质量量化指标）：
  - Hit Rate@K：N 个测试问题中，黄金答案块（gold chunk）出现在检索 Top-K 里的比例
  - MRR（Mean Reciprocal Rank）：第一个命中块的排名倒数的平均值

默认用 RRF 融合结果评估（不调 LLM，稳定、快、0 成本）；
加 --rerank 则走完整 retrieve（含 LLM ReRank，会消耗 DeepSeek 额度）。

用法（项目根目录）：
    & .venv\\Scripts\\python.exe eval_retrieval.py
    & .venv\\Scripts\\python.exe eval_retrieval.py --rerank
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import config
from app.rag import RAGEngine

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------
# 测试集：(问题, 黄金答案片段)
#   命中判定：该片段必须出现在被检索到的文档块文本中（子串匹配）。
#   覆盖文档 7 大章节：攀爬系统 / 难度曲线 / 物理系统 / 视觉 / 音效 / 收集成就 / 概述。
# ---------------------------------------------------------------
TEST_CASES = [
    ("攀爬系统有几个状态？", "共有 6 个状态"),
    ("角色跳跃的高度是多少？", "跳跃高度：2.5m"),
    ("游戏里的重力加速度是多少？", "重力加速度：-15 m/s²"),
    ("教程区的平台宽度是多少？", "平台宽度：3m"),
    ("地狱区在哪里设置检查点？", "600m 和 750m"),
    ("OnlyUp! 用什么引擎开发？", "Unity 2022 LTS"),
    ("角色冲刺速度是多少？", "冲刺 12 m/s"),
    ("墙壁上的攀爬速度是多少？", "3 m/s（墙壁）"),
    ("角色用什么碰撞体？", "Capsule Collider"),
    ("收集金色羽毛有什么奖励？", "全部收集解锁皮肤"),
    ("登顶者成就的要求是什么？", "到达 1000m"),
    ("完美主义者成就要求什么？", "收集全部金色羽毛"),
    ("最大坠落速度是多少？", "-30 m/s"),
    ("空气阻力系数是多少？", "0.02"),
    ("检查点激活时播放什么音效？", "水晶碎裂声"),
    ("普通跳跃和蓄力跳跃的初速度？", "12 m/s（蓄力）"),
    ("攀爬时体力每秒消耗多少？", "每秒减少 3 点体力"),
    ("热身区在多少米之间？", "50m ~ 200m"),
    ("时间限制区域在哪一阶段出现？", "时间限制区域"),
    ("隐形平台有什么特性？", "隐形平台（靠近才显示）"),
]


def resolve_gold_indices(engine: RAGEngine) -> dict[str, set[int]]:
    """把每个问题的黄金答案片段匹配到文档块索引（子串匹配）。"""
    gold: dict[str, set[int]] = {}
    for _q, answer_key in TEST_CASES:
        hits = {
            i for i, doc in enumerate(engine.all_documents) if answer_key in doc
        }
        gold[answer_key] = hits
    return gold


def hit_rate_stats(gold: set[int], ranked_indices: list[int]) -> dict:
    """对单条查询计算 @1/@3/@5 命中与 MRR。"""
    hit_at = {k: False for k in (1, 3, 5)}
    mrr = 0.0
    for rank, idx in enumerate(ranked_indices, start=1):
        if idx in gold:
            if rank == 1:
                hit_at[1] = True
            if rank <= 3:
                hit_at[3] = True
            if rank <= 5:
                hit_at[5] = True
            mrr = 1.0 / rank
            break
    return {"hit_at": hit_at, "mrr": mrr}


def _eval_mode(engine: RAGEngine, gold_map: dict, use_rerank: bool) -> tuple[list[dict], dict]:
    """按指定模式评估，返回 (rows, summary)。use_rerank=False → RRF 融合（无 LLM）。"""
    rows = []
    agg = {"hits": {1: 0, 3: 0, 5: 0}, "mrr": 0.0}
    for i, (query, answer_key) in enumerate(TEST_CASES, start=1):
        gold = gold_map[answer_key]
        if use_rerank:
            results = engine.retrieve(query, top_k=5)
            ranked = [r["index"] for r in results]
        else:
            bm25 = engine._bm25_search(query)
            sem = engine._semantic_search(query)
            fused = engine._reciprocal_rank_fusion(bm25, sem)
            ranked = [r["index"] for r in fused[:5]]

        stats = hit_rate_stats(gold, ranked)
        for k in (1, 3, 5):
            agg["hits"][k] += int(stats["hit_at"][k])
        agg["mrr"] += stats["mrr"]

        # 记录每个返回块的文本摘要（前 80 字符），用于 md 明细
        top_texts = []
        for idx in ranked:
            snippet = engine.all_documents[idx] if idx < len(engine.all_documents) else ""
            top_texts.append(snippet[:80].replace("\n", " "))
        hit_rank = next(
            (rank for rank, idx in enumerate(ranked, start=1) if idx in gold), None
        )

        rows.append(
            {
                "no": i,
                "query": query,
                "answer_key": answer_key,
                "gold": sorted(gold),
                "top": ranked,
                "top_texts": top_texts,
                "hit_rank": hit_rank,
                "hit_at": stats["hit_at"],
                "mrr": round(stats["mrr"], 3),
            }
        )
    n = len(TEST_CASES)
    summary = {
        "total": n,
        "hit_rate@1": round(agg["hits"][1] / n, 4),
        "hit_rate@3": round(agg["hits"][3] / n, 4),
        "hit_rate@5": round(agg["hits"][5] / n, 4),
        "mrr": round(agg["mrr"] / n, 4),
        "mode": "RRF + LLM ReRank" if use_rerank else "RRF 融合",
    }
    return rows, summary


def run(use_rerank: bool = False, embed: str = "onnx") -> None:
    print("=" * 76)
    print("📊 OnlyUp! RAG 检索命中率评估")
    print("=" * 76)

    # bge 中文模型与 onnx 英文模型维度不同，需要隔离的 Chroma 库
    if embed == "bge":
        chroma_dir = config.CHROMA_DIR_BGE
    else:
        chroma_dir = config.CHROMA_DIR

    engine = RAGEngine(chroma_dir=chroma_dir, embedding_model=embed)
    engine.build()
    print(f"📚 知识库：{len(engine.all_documents)} 个文档块（embedding={embed}）")

    gold_map = resolve_gold_indices(engine)
    # 校验：每个黄金片段必须至少匹配到 1 个块，否则测试集本身有问题
    missing = [k for k, g in gold_map.items() if not g]
    if missing:
        print(f"⚠️ 以下黄金片段未匹配到任何块（请修正测试集）：{missing}")
    print(f"✅ 测试集：{len(TEST_CASES)} 个问题（含黄金答案块匹配）\n")

    rrf_rows, rrf_sum = _eval_mode(engine, gold_map, use_rerank=False)
    if use_rerank:
        rr_rows, rr_sum = _eval_mode(engine, gold_map, use_rerank=True)
    else:
        rr_rows, rr_sum = None, None

    def _print_rows(title: str, rows: list[dict]) -> None:
        print("-" * 76)
        print(f"📋 {title}")
        print("-" * 76)
        for r in rows:
            mark = "✅" if r["hit_at"][3] else ("◐" if r["hit_at"][5] else "❌")
            print(
                f"[{r['no']:02d}] {mark} Q: {r['query']}\n"
                f"      金块={r['gold']}  命中@1/3/5="
                f"{int(r['hit_at'][1])}/{int(r['hit_at'][3])}/{int(r['hit_at'][5])}  "
                f"MRR={r['mrr']:.3f}"
            )
        print()

    def _print_summary(summary: dict) -> None:
        print(f"  Hit Rate@1 = {summary['hit_rate@1']:.2%}  ({summary['total'] * summary['hit_rate@1']:.0f}/{summary['total']})")
        for k in (3, 5):
            print(f"  Hit Rate@{k} = {summary[f'hit_rate@{k}']:.2%}  ({summary['total'] * summary[f'hit_rate@{k}']:.0f}/{summary['total']})")
        print(f"  MRR          = {summary['mrr']:.3f}")

    _print_rows("RRF 融合（无 LLM，检索引擎本身）", rrf_rows)
    _print_summary(rrf_sum)

    if rr_rows is not None:
        _print_rows("RRF + LLM ReRank（端到端）", rr_rows)
        _print_summary(rr_sum)

        print("\n" + "=" * 76)
        print("📈 模式对比")
        print("=" * 76)
        for k in (1, 3, 5):
            a, b = rrf_sum[f"hit_rate@{k}"], rr_sum[f"hit_rate@{k}"]
            delta = "+" if b > a else ("-" if b < a else "=")
            print(f"  Hit Rate@{k}  RRF={a:.2%}  ReRank={b:.2%}  ({delta})")
        print(f"  MRR      RRF={rrf_sum['mrr']:.3f}  ReRank={rr_sum['mrr']:.3f}")

    # 写出评估报告（Markdown，简历素材）——按 embedding 类型隔离文件名
    tag = "BGE" if embed == "bge" else "ONNX"
    out = render_report(rrf_sum, rr_sum, rrf_rows, rr_rows, gold_map, embed=embed)
    out_path = Path(__file__).resolve().parent / "docs" / f"RETRIEVAL_EVAL_{tag}.md"
    out_path.write_text(out, encoding="utf-8")
    print(f"\n📄 报告已写入：{out_path}")

    # 同时输出 JSON（便于程序化处理）
    json_path = Path(__file__).resolve().parent / "docs" / f"retrieval_eval_{tag.lower()}.json"
    json_path.write_text(
        json.dumps(
            {"embed": embed, "rrf": {"summary": rrf_sum, "cases": rrf_rows},
             "rerank": {"summary": rr_sum, "cases": rr_rows} if rr_sum else None},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"📄 数据已写入：{json_path}")


def render_report(
    rrf_sum: dict, rr_sum: dict | None, rrf_rows: list[dict], rr_rows: list[dict] | None,
    gold_map: dict, embed: str = "onnx",
) -> str:
    embed_name = "BAAI/bge-small-zh-v1.5（中文）" if embed == "bge" else "ONNXMiniLM_L6_V2（英文）"
    lines = [
        "# 检索质量评估报告（Retrieval Hit Rate）",
        "",
        f"- Embedding 模型：**{embed_name}**",
        f"- 测试样本：**{rrf_sum['total']}** 个问题（覆盖 GDD 七大章节）",
        "- 判定标准：黄金答案片段出现在检索 Top-K 中即视为命中",
        "",
        "## 汇总指标",
        "",
        "| 指标 | RRF 融合 | " + ("RRF + LLM ReRank |" if rr_sum else "|"),
        "| --- | --- " + ("| --- |" if rr_sum else "|"),
    ]
    for k in (1, 3, 5):
        rr_cell = f"| **{rr_sum[f'hit_rate@{k}']:.2%}** |" if rr_sum else "|"
        lines.append(f"| Hit Rate@{k} | **{rrf_sum[f'hit_rate@{k}']:.2%}** {rr_cell}")
    rr_cell = f"| **{rr_sum['mrr']:.3f}** |" if rr_sum else "|"
    lines.append(f"| MRR | **{rrf_sum['mrr']:.3f}** {rr_cell}")
    lines += _render_detail("RRF 融合", rrf_rows)
    if rr_rows:
        lines += _render_detail("RRF + LLM ReRank", rr_rows)
    lines += [
        "",
        "## 说明",
        "",
        "- 命中判定使用黄金答案片段与文档块的子串匹配（gold chunk 可能因重叠跨多个块）。",
        "- 默认评估 RRF 融合后的原始排序，反映**检索引擎本身**的质量；ReRank 只做最终微调。",
    ]
    return "\n".join(lines)


def _render_detail(title: str, rows: list[dict]) -> list[str]:
    """渲染某个模式的逐题明细 + 每问 Top-5 返回块内容摘要。"""
    lines = ["", f"## 逐题明细（{title}）", "", "| # | 问题 | 命中@1 | 命中@3 | 命中@5 | MRR |", "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(
            f"| {r['no']} | {r['query']} | {'✅' if r['hit_at'][1] else '—'} | "
            f"{'✅' if r['hit_at'][3] else '—'} | {'✅' if r['hit_at'][5] else '—'} | {r['mrr']:.3f} |"
        )
    lines += ["", "### 检索返回明细", ""]
    for r in rows:
        gold_str = ", ".join(f"`{g}`" for g in r["gold"]) or "无匹配"
        hit_mark = (
            f"第 {r['hit_rank']} 位命中 ✅" if r["hit_rank"] else "Top-5 内未命中 ❌"
        )
        lines.append(
            f"**{r['no']:02d}. {r['query']}**  "
            f"（gold={gold_str}，{hit_mark}，MRR={r['mrr']:.3f}）"
        )
        lines.append("")
        lines.append("| 排名 | 块 | 内容摘要 |")
        lines.append("| --- | --- | --- |")
        for rank, (idx, text) in enumerate(zip(r["top"], r["top_texts"]), start=1):
            is_gold = "⭐" if idx in r["gold"] else "　"
            lines.append(f"| {rank} | `#{idx}` {is_gold} | {text} |")
        lines.append("")
    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OnlyUp! RAG 检索命中率评估")
    parser.add_argument(
        "--rerank", action="store_true", help="走完整 retrieve（含 LLM ReRank，消耗 DeepSeek）"
    )
    parser.add_argument(
        "--embed", choices=["onnx", "bge"], default="onnx",
        help="embedding 模型：onnx=英文 ONNX（默认），bge=中文 BAAI/bge-small-zh-v1.5",
    )
    args = parser.parse_args()
    run(use_rerank=args.rerank, embed=args.embed)
