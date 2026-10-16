# -*- coding: utf-8 -*-
r"""
第 6 步：命令行问答 / 批量出题（做实验记录用）
==============================================
网页是给人看的，这个脚本是给「逐题记录」用的：把一批问题跑一遍，
把**召回了哪些块、回答是什么、用了多少 token** 全部落盘成 JSON，
后面写实验记录和结论都从这里取数，不靠回忆。

用法：
    # 单问一句
    python scripts/06_ask.py --q "美的集团上半年货币资金是多少"

    # 批量跑问题清单（问题清单是 json 数组，见 data/questions.json）
    python scripts/06_ask.py --file data/questions.json --mode hybrid \
        --out results/qa_hybrid.json

    # 四种检索模式各跑一遍同一批题（对照实验）
    python scripts/06_ask.py --file data/questions.json --mode all \
        --out results/qa_all.json
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import retrieval                                   # noqa: E402


def run_one(store, question, mode, top_k, rewriter, group=False, keywordizer=None):
    t0 = time.time()
    hits = store.search(question, mode=mode, top_k=top_k, rewriter=rewriter,
                        company_aware=group, keywordizer=keywordizer)
    from retrieval import build_answer_prompt, ANSWER_SYSTEM, llm_chat
    ans, usage = llm_chat([
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": build_answer_prompt(question, hits)}])
    return {
        "question": question, "mode": mode, "seconds": round(time.time() - t0, 1),
        "answer": ans, "usage": usage,
        "hits": [{"rank": h["rank"], "id": h["chunk"]["id"], "company": h["chunk"]["company"],
                  "section": h["chunk"]["section"], "page": h["chunk"]["page"],
                  "kind": h["chunk"]["kind"], "sim": h["sim"], "bm25": h["bm25"],
                  "score": h["score"], "text": h["chunk"]["raw"][:1200]} for h in hits],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", help="单个问题")
    ap.add_argument("--file", help="问题清单 json：[{\"id\":1,\"q\":\"...\",\"type\":\"单公司\"}]")
    ap.add_argument("--mode", default="hybrid",
                    choices=["vector", "bm25", "hybrid", "rewrite", "kw", "all"])
    ap.add_argument("--group", action="store_true", help="启用按公司分组召回（跨公司题）")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--out", help="结果落盘路径（json）")
    args = ap.parse_args()

    store = retrieval.Store()
    print(f"索引就绪：{len(store.chunks)} 块 | 向量 {'有' if store.emb is not None else '缺'} | "
          f"BM25 {'有' if store.bm25 is not None else '缺'}")
    rewriter = retrieval.make_rewriter() if args.mode in ("rewrite", "all") else None
    kwizer = retrieval.make_keyword_extractor() if args.mode in ("kw", "all") else None

    if args.q:
        r = run_one(store, args.q, args.mode if args.mode != "all" else "hybrid",
                    args.top_k, rewriter, args.group, kwizer)
        print("\n【回答】\n" + r["answer"])
        print("\n【召回】")
        for h in r["hits"]:
            print(f"  [{h['rank']}] {h['company']} {h['section']} p{h['page']} "
                  f"({h['kind']}) sim={h['sim']} bm25={h['bm25']}")
        return

    qs = [json.loads(l) if l.strip().startswith("{") else None
          for l in open(args.file, encoding="utf-8")] if False else \
        json.load(open(args.file, encoding="utf-8"))
    modes = ["vector", "bm25", "hybrid", "rewrite", "kw"] if args.mode == "all" else [args.mode]

    all_res = []
    for mode in modes:
        for item in qs:
            q = item["q"] if isinstance(item, dict) else item
            rec = run_one(store, q, mode, args.top_k, rewriter, args.group, kwizer)
            rec["qid"] = item.get("id") if isinstance(item, dict) else None
            rec["qtype"] = item.get("type") if isinstance(item, dict) else None
            all_res.append(rec)
            print(f"[{mode:8s}] #{rec['qid']} {rec['seconds']:5.1f}s  "
                  f"{q[:38]}", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(all_res, open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"\n{len(all_res)} 条记录 -> {args.out}")


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                          # noqa: BLE001
            pass
    main()
