# -*- coding: utf-8 -*-
r"""
第 8 步：给 10 道题标「标准答案块」，量一量召回
================================================
「答对没有」这句话要有依据，不能靠感觉。做法：先把每道题的答案位置人工标出来
（哪家公司的哪一页/哪一块里有答案），再让四种检索模式各召回一次，看：

    · 第一个标准答案块排第几（first_rank，越小越好）
    · 前 8 块里有没有标准答案（hit@8）
    · 前 8 块覆盖了几家公司（跨公司题必须覆盖多家，否则答不全）

标准答案块是**人读原文标出来的**（GROUND_TRUTH 里的 公司+页码+关键词），
不是模型生成的——这样评测才独立于被测系统。

用法：
    python scripts/08_eval_retrieval.py            # 四种模式全跑
    python scripts/08_eval_retrieval.py --out results/recall_eval.json
"""
import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import retrieval                                   # noqa: E402

# ---- 标准答案：人读原文标出来的 ----
# 标法：不标「我看的那一页」，而是标**所有含答案数值的块**——
# 同一笔钱在一份半年报里往往出现在三处（利润表、费用分析表、附注），
# 只要召回到其中任何一处，模型就能答对，那就该算命中。
# 每条 = (公司, 答案里必须出现的一个字符串)
GROUND_TRUTH = {
    1: [("美的集团", "90,642,783")],                       # 货币资金（千元）
    2: [("格力电器", "89,397,522,501.21"), ("格力电器", "13,277,590,882.57")],
    3: [("海尔智家", "9,752,033,977.71"), ("海尔智家", "主要系销售收入下降")],
    4: [("石头科技", "4,979,861,715")],                     # 境外资产
    5: [("新宝股份", "136,439,441.49")],                    # 归母净利润
    6: [("科沃斯", "1,247,851,012.57"), ("科沃斯", "49.0"), ("科沃斯", "5.54")],
    7: [("美的集团", "21,545,703"), ("格力电器", "5,129,627,933.54"),
        ("海尔智家", "16,438,747,513.12")],                # 三家销售费用
    8: [],                                                 # 全景题：不定块，只看覆盖公司数
    9: [("美的集团", "以旧换新"), ("小熊电器", "以旧换新")],
    10: [("美的集团", "260,042,490"), ("格力电器", "89,397,522,501.21"),
         ("海尔智家", "152,115,050,237.63")],              # 三家营业收入
}


def gold_chunk_ids(store):
    """把 (公司, 答案里必须出现的字符串) 映射到具体的块 id 集合。"""
    gold = {}
    for qid, specs in GROUND_TRUTH.items():
        ids = []
        for comp, kw in specs:
            for c in store.chunks:
                if c["company"] == comp and kw in c["raw"]:
                    ids.append(c["id"])
        gold[qid] = sorted(set(ids))
    return gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "recall_eval.json"))
    args = ap.parse_args()

    store = retrieval.Store()
    questions = json.load(open(os.path.join(ROOT, "data", "questions.json"), encoding="utf-8"))
    gold = gold_chunk_ids(store)

    print(f"语料 {len(store.chunks)} 块 | 10 道题的标准答案块已定位")
    missing = {q: len(v) for q, v in gold.items() if not v}
    if missing:
        print("⚠️ 有题没标到标准答案块（检查 GROUND_TRUTH）:", missing)

    out = []
    for item in questions:
        qid = item["id"]
        row = {"id": qid, "type": item["type"], "q": item["q"],
               "gold_ids": gold[qid], "modes": {}}
        print(f"\n=== Q{qid} [{item['type']}] {item['q'][:44]}")
        rewriter = retrieval.make_rewriter()
        kwizer = retrieval.make_keyword_extractor()
        for mode in ("vector", "bm25", "hybrid", "rewrite", "hybrid+分组", "抽词+分组"):
            base_mode = {"hybrid+分组": "hybrid", "抽词+分组": "kw"}.get(mode, mode)
            hits = store.search(item["q"], mode=base_mode, top_k=args.top_k,
                                company_aware=mode in ("hybrid+分组", "抽词+分组"),
                                rewriter=rewriter, keywordizer=kwizer)
            ids = [h["chunk"]["id"] for h in hits]
            first = next((i + 1 for i, x in enumerate(ids) if x in gold[qid]), None)
            hit = first is not None
            comps = []
            for h in hits:
                if h["chunk"]["company"] not in comps:
                    comps.append(h["chunk"]["company"])
            row["modes"][mode] = {
                "first_gold_rank": first, "hit": hit, "n_companies": len(comps),
                "companies": comps, "ids": ids,
                "top": [{"rank": h["rank"], "company": h["chunk"]["company"],
                         "section": h["chunk"]["section"], "page": h["chunk"]["page"],
                         "kind": h["chunk"]["kind"], "sim": h["sim"], "bm25": h["bm25"],
                         "is_gold": h["chunk"]["id"] in gold[qid]} for h in hits],
            }
            print(f"   {mode:8s} 首个答案块排名={str(first):>5s} 命中={hit} "
                  f"覆盖公司数={len(comps)}  {comps[:4]}")
        out.append(row)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n-> {args.out}")

    # 汇总表
    print("\n模式       命中@8   平均首个答案块排名")
    for mode in ("vector", "bm25", "hybrid", "rewrite", "hybrid+分组", "抽词+分组"):
        hit = sum(1 for r in out if r["modes"][mode]["hit"])
        ranks = [r["modes"][mode]["first_gold_rank"] for r in out if r["modes"][mode]["first_gold_rank"]]
        print(f"{mode:12s} {hit:>4}/10   {sum(ranks)/len(ranks):.1f}" if ranks else f"{mode:12s} {hit:>4}/10")


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                          # noqa: BLE001
            pass
    main()
