# -*- coding: utf-8 -*-
r"""
第 9 步：文本块星空图（给页面配图）
===================================
把 5876 个块的 1024 维向量压到二维，让「一座小岛 = 一家公司」看得见。
做法和课件一致：先用 PCA 降到 50 维（去噪声、提速），再用 t-SNE 压到 2 维。

课件用的是 UMAP，这里用 sklearn 的 t-SNE，理由很实际：
umap-learn 依赖 numba，numba 对 numpy 版本很挑，装不上就整条链跑不动；
sklearn 的 TSNE 是纯编译扩展，装完就能用，出来的图同样是「一岛一家公司」。

产物：data/map.json  {"points": [[x, y, 公司序号, 类型, 块id], ...], "colors": [...], "companies": [...]}
用法：python scripts/09_make_map.py
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "map.json")

COLORS = ["#1f6feb", "#7a4fd6", "#137a4b", "#a35a00", "#b42318", "#0e7490", "#9333ea",
          "#4d7c0f", "#be185d", "#0369a1", "#b45309", "#15803d", "#6d28d9", "#9f1239", "#0f766e"]


def main():
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    chunks = [json.loads(l) for l in open(os.path.join(ROOT, "data", "chunks.jsonl"), encoding="utf-8")]
    emb = np.load(os.path.join(ROOT, "data", "emb.npy"))
    assert len(chunks) == emb.shape[0]

    companies = sorted({c["company"] for c in chunks})
    comp_idx = {c: i for i, c in enumerate(companies)}

    t0 = time.time()
    z = PCA(n_components=50, random_state=0).fit_transform(emb)
    print(f"PCA 1024->50 完成 {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=0,
              metric="cosine", n_jobs=os.cpu_count()).fit_transform(z)
    print(f"t-SNE 50->2 完成 {time.time()-t0:.1f}s", flush=True)

    pts = [[round(float(xy[i, 0]), 3), round(float(xy[i, 1]), 3),
            comp_idx[chunks[i]["company"]], chunks[i]["kind"], chunks[i]["id"]]
           for i in range(len(chunks))]
    json.dump({"points": pts, "colors": COLORS[:len(companies)], "companies": companies},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"-> {OUT}（{len(pts)} 个点，{len(companies)} 家公司）")


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                          # noqa: BLE001
            pass
    main()
