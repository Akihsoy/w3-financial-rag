# -*- coding: utf-8 -*-
r"""
第 4 步：把每个块编码成向量（本地 Qwen3-Embedding-0.6B）
========================================================
为什么用本地模型：财报里没有保密内容，但「公司的库不该出本机」是个好习惯；
更重要的是本地方案不要 key、不按量计费，6704 个块一次算完，之后随时重建。

课件口径：本机 Qwen3-Embedding-0.6B，1024 维，查询侧加指令前缀。

工程上三个坑，脚本里都处理了：
  ① CPU 跑 0.6B 模型慢得离谱（实测 0.6 块/秒 ≈ 3 小时）——有 N 卡就用 N 卡；
  ② 中途断了要能续跑——每 256 块往盘上存一次，重跑自动跳过已算的；
  ③ 查询和文档要用**两套编码方式**：文档原样编码，查询要带 Instruct 前缀。

产物：
    data/emb.npy             (n_chunks, 1024) float32，行序与 chunks.jsonl 一致
    data/emb_meta.json       模型名/维度/块数/耗时

用法：
    python scripts/04_embed.py            # 自动选 GPU/CPU
    python scripts/04_embed.py --cpu      # 强制 CPU
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_JSONL = os.path.join(ROOT, "data", "chunks.jsonl")
OUT_NPY = os.path.join(ROOT, "data", "emb.npy")
OUT_META = os.path.join(ROOT, "data", "emb_meta.json")
PART_NPY = os.path.join(ROOT, "data", "emb.part.npy")

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
BATCH = 64
SAVE_EVERY = 256

# 查询侧指令前缀：Qwen3-Embedding 官方要求，检索任务实测再涨 1–5%
QUERY_INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"


def load_model(device):
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"模型加载完成 {time.time()-t0:.1f}s | device={device} | "
          f"dim={model.get_sentence_embedding_dimension()} | max_len={model.max_seq_length}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="强制用 CPU")
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()

    import torch
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), "| fp16 推理")
    else:
        torch.set_num_threads(os.cpu_count() or 8)

    chunks = [json.loads(l) for l in open(IN_JSONL, encoding="utf-8")]
    texts = [c["embed_text"] for c in chunks]
    n = len(texts)

    # ---- 按长度排序再分批：同一批长度接近，padding 少、显存省、速度快 ----
    # （不排序时，一个 2000 字的表格块会把整批 64 块都撑到 3500 token，
    #   6GB 显存上必爆 OOM —— 这是实测踩出来的）
    order = sorted(range(n), key=lambda i: len(texts[i]))

    # ---- 断点续跑：emb.part.npy 里已经算好的行不再重算 ----
    done, vecs_done = 0, []
    if os.path.exists(PART_NPY):
        arr = np.load(PART_NPY)
        done = arr.shape[0]
        vecs_done = list(arr)
        print(f"发现未完成结果：已完成 {done}/{n} 块，从这里继续")

    model = load_model(device)
    if device == "cuda":
        model = model.half()                            # 半精度，显存减半、速度翻倍

    t0 = time.time()
    out = [None] * n
    for j in range(done):
        out[order[j]] = vecs_done[j]
    cur = args.batch
    i = done
    with torch.inference_mode():
        while i < n:
            idx = order[i:i + cur]
            try:
                emb = model.encode([texts[k] for k in idx], batch_size=len(idx),
                                   normalize_embeddings=True, convert_to_numpy=True,
                                   show_progress_bar=False)
            except RuntimeError as e:                   # 显存不够：清缓存、批减半、原地重试
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                cur = max(1, cur // 2)
                print(f"  [OOM] {str(e)[:60]}… 批大小降到 {cur} 重试", flush=True)
                if cur == 1:
                    raise
                continue
            for k, v in zip(idx, emb):
                out[k] = v.astype(np.float32)
            i += len(idx)
            if i - done >= SAVE_EVERY or i >= n:
                np.save(PART_NPY, np.asarray([out[k] for k in order[:i]], dtype=np.float32))
                sp = (i - done) / max(time.time() - t0, 1e-6)
                eta = (n - i) / max(sp, 1e-6)
                print(f"  {i:>5}/{n} 块 | {sp:5.1f} 块/秒 | 预计还需 {eta/60:4.1f} 分钟", flush=True)

    emb = np.asarray(out, dtype=np.float32)
    assert emb.shape == (n, 1024), f"形状不对: {emb.shape}"
    np.save(OUT_NPY, emb)
    os.remove(PART_NPY)
    meta = {"model": MODEL_NAME, "dim": int(emb.shape[1]), "n_chunks": n,
            "device": device, "seconds": round(time.time() - t0, 1),
            "query_prefix": QUERY_INSTRUCT}
    json.dump(meta, open(OUT_META, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n完成 {emb.shape}，用时 {(time.time()-t0)/60:.1f} 分钟 -> {OUT_NPY}")


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                            # noqa: BLE001
            pass
    main()
