# -*- coding: utf-8 -*-
r"""
第 11 步：生成可以放 Vercel 的静态演示站
=========================================
为什么不能直接把整套系统搬上 Vercel：
    本地问答依赖 ①1.2 GB 的向量模型（torch + Qwen3-Embedding）
    ②numpy 索引与 jieba 倒排 ③带 key 的 DeepSeek 调用。
    这三样在 Vercel 的静态托管里都跑不了（没有常驻进程、函数也有体积上限），
    硬塞就得把 key 放到公网服务里，等于把钱包挂在网上。

所以这里生成的是**静态演示站**：把本地真实跑出来的结果（10 道题 × 2 种检索模式
× 召回明细 + 召回评测 + 星空图）固化成一堆 json，页面只做展示和回放，
不联网、不需要 key。要真·在线提问，得自己加后端（见 site/README_部署.md）。

产物（都放仓库根目录，Vercel / GitHub Pages 零配置即可打开）：
    index.html      页面（手写，随仓库一起版本管理）
    demo.json       10 道题的回答 + 召回明细 + 判卷 + 召回评测
    map.json        星空图坐标（5876 个点，带 140 字预览）
    assets/         一页结论 PDF 与两张截图

用法：python scripts/11_build_site.py
"""
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = ROOT                     # 页面和它的数据直接放仓库根目录（Vercel 零配置就能跑）
ASSETS = os.path.join(SITE, "assets")

PREVIEW_CHARS = 140          # 星空图里每个块带的预览字数


def load(path, default=None):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return default
    return json.load(open(p, encoding="utf-8"))


def main():
    os.makedirs(ASSETS, exist_ok=True)

    questions = load("data/questions.json", [])
    hybrid = {r["qid"]: r for r in load("results/qa_hybrid.json", [])}
    kwgroup = {r["qid"]: r for r in load("results/qa_kw_group.json", [])}
    recall = {r["id"]: r for r in load("results/recall_eval.json", [])}
    verdict = load("results/判卷.json", {})
    chunks = {}
    for line in open(os.path.join(ROOT, "data", "chunks.jsonl"), encoding="utf-8"):
        c = json.loads(line)
        chunks[c["id"]] = c

    def pack_hits(r):
        return [{"rank": h["rank"], "id": h["id"], "company": h["company"],
                 "section": h["section"], "page": h["page"], "kind": h["kind"],
                 "sim": round(h["sim"], 3), "bm25": round(h["bm25"], 1),
                 "text": h["text"]} for h in r["hits"]]

    items = []
    for q in questions:
        qid = str(q["id"])
        e = recall.get(q["id"], {})
        rc = {}
        for m, d in (e.get("modes") or {}).items():
            rc[m] = {"first": d.get("first_gold_rank"), "hit": d.get("hit"),
                     "ncomp": d.get("n_companies"), "companies": d.get("companies", [])[:6]}
        modes = {}
        if q["id"] in hybrid:
            r = hybrid[q["id"]]
            modes["hybrid"] = {"label": "基本混合（向量+BM25）", "seconds": r["seconds"],
                               "tokens": (r.get("usage") or {}).get("total_tokens"),
                               "answer": r["answer"], "hits": pack_hits(r)}
        if q["id"] in kwgroup:
            r = kwgroup[q["id"]]
            modes["kw"] = {"label": "抽词+分组（跨公司题的解法）", "seconds": r["seconds"],
                           "tokens": (r.get("usage") or {}).get("total_tokens"),
                           "answer": r["answer"], "hits": pack_hits(r)}
        v = verdict.get(qid, {})
        items.append({
            "id": q["id"], "type": q["type"], "q": q["q"],
            "verdict": v.get("verdict", ""), "right": v.get("right", ""),
            "wrong": v.get("wrong", ""), "why": v.get("why", ""),
            "gold": e.get("gold_ids", []), "recall": rc, "modes": modes,
        })

    meta = load("data/emb_meta.json", {})
    stats = {
        "n_chunks": len(chunks),
        "n_text": sum(1 for c in chunks.values() if c["kind"] == "text"),
        "n_table": sum(1 for c in chunks.values() if c["kind"] == "table"),
        "n_companies": len({c["company"] for c in chunks.values()}),
        "model": meta.get("model", "Qwen/Qwen3-Embedding-0.6B"),
        "dim": meta.get("dim", 1024),
        "pages": 2588, "chars_wan": 310, "tables": 3852,
        "generated": "2026-10-16",
    }
    json.dump({"stats": stats, "questions": items},
              open(os.path.join(SITE, "demo.json"), "w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))
    print(f"demo.json  {len(items)} 道题")

    # 星空图：坐标 + 公司序号 + 类型 + 块 id + 预览
    m = load("data/map.json", {"points": [], "companies": [], "colors": []})
    pts = []
    for x, y, ci, kind, cid in m["points"]:
        c = chunks.get(cid)
        pts.append([round(x, 1), round(y, 1), ci, 1 if kind == "table" else 0, cid,
                    (c["raw"][:PREVIEW_CHARS] if c else "")])
    json.dump({"companies": m["companies"], "colors": m["colors"], "points": pts},
              open(os.path.join(SITE, "map.json"), "w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))
    print(f"map.json   {len(pts)} 个点")

    # 资产
    for src, dst in [("results/一页结论.pdf", "一页结论.pdf"),
                     ("截图/结果_截图1_答对带出处.png", "截图1_答对带出处.png"),
                     ("截图/结果_截图2_跨公司题翻车.png", "截图2_跨公司题翻车.png")]:
        p = os.path.join(ROOT, src)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(ASSETS, dst))
            print(f"assets/{dst}")
        else:
            print(f"  [跳过] 缺 {src}")

    print(f"\n完成：{SITE}")
    print("本地预览：在本目录跑 python -m http.server 8080  -> http://127.0.0.1:8080")


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                          # noqa: BLE001
            pass
    main()
