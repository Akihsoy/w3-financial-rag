# -*- coding: utf-8 -*-
r"""
第 5 步：建 BM25 倒排索引
=========================
课件口径：BM25 用 jieba 分词建倒排（课件全库 703,012 块、216 秒、词表 67.6 万）。

BM25 的三个直觉（课件语）：
    ① 一个词在这块里出现次数越多，这块越相关；
    ② 一个词在全库越少见，它越值钱（IDF）——「货币资金」比「公司」值钱得多；
    ③ 块越长越吃亏（长度归一化），否则长块光靠字多就能赢。

为什么财报库一定要有它：科目名、公司名、人名、年份、条款号都是「字面必须出现」
的东西，而口语化提问（「上半年赚了多少钱」）正是向量的强项——两条各管一段。

产物：data/bm25.pkl
用法：python scripts/05_index_bm25.py
"""
import json
import os
import pickle
import sys
import time

import re

import jieba

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_JSONL = os.path.join(ROOT, "data", "chunks.jsonl")
OUT_PKL = os.path.join(ROOT, "data", "bm25.pkl")


def tokenize(text):
    """jieba 精确模式分词；丢掉空白与纯标点，保留中文词、数字、英文单词。

    关键一步：先把千分位逗号去掉。财报里的数都是「90,642,783」这种写法，
    不处理的话 jieba 会把它切成 '90' ',' '642' ',' '783' 四段，
    用户问「90642783」或者问完整数值时 BM25 完全对不上——
    而财报检索里，数字对不上几乎是致命的。
    """
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    out = []
    for t in jieba.lcut(text):
        t = t.strip()
        if not t:
            continue
        # 丢纯标点：注意不能只判 isalnum——「5129627933.54」这种带小数点的
        # 数字会被判成「非字母数字」而整段丢掉，财报检索里这是致命的
        # （实测：加上这条判断前，全库所有带逗号/小数点的金额都进不了索引）
        if t.isascii() and not t.isalnum() and not any(ch.isdigit() for ch in t):
            continue
        out.append(t)
    return out


def main():
    from rank_bm25 import BM25Okapi
    chunks = [json.loads(l) for l in open(IN_JSONL, encoding="utf-8")]
    t0 = time.time()
    tokens = []
    for i, c in enumerate(chunks):
        tokens.append(tokenize(c["embed_text"]))
        if (i + 1) % 2000 == 0:
            print(f"  分词 {i+1}/{len(chunks)} … {time.time()-t0:.0f}s", flush=True)
    vocab = set(t for doc in tokens for t in doc)
    bm25 = BM25Okapi(tokens)
    with open(OUT_PKL, "wb") as f:
        pickle.dump({"bm25": bm25, "tokens": tokens}, f)
    print(f"完成：{len(chunks)} 块 | 词表 {len(vocab):,} | 用时 {time.time()-t0:.0f}s -> {OUT_PKL}")

    # 自检：随手搜一个词，看命中的是不是该出现的块
    import numpy as np
    for q in ["合同负债", "石头科技 海外收入"]:
        qs = tokenize(q)
        sc = bm25.get_scores(qs)
        top = np.argsort(-sc)[:3]
        print(f"\n自检『{q}』top3：")
        for i in top:
            c = chunks[int(i)]
            print(f"   {sc[i]:6.2f}  {c['company']} {c['section']} p{c['page']}")


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                            # noqa: BLE001
            pass
    main()
