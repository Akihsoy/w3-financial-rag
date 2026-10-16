# -*- coding: utf-8 -*-
r"""
第 3 步：切块（chunking）
=========================
按课件的口径切：**800 字一块、相邻块重叠 120 字、尽量在句末收尾**。

在课件基础上多做两件事，都是为了让「出处」站得住：
  ① 四个标签：公司 / 章节 / 页码 / 块序号 —— 每个块都带，展示和溯源都要用；
  ② 向量化文本 = 「公司名 · 章节名 + 原文」，把归属写进被编码的文字里
     （课件原话：向量化时把「公司名 + 章节名」一起放进文本，检索时才分得清是谁的数）。

表格单独处理：
  表格按行切开，但**每一块都带上表头和表名**——课件实验一里伊利那道题翻车，
  就是因为「表格切开以后，表名留在了前面的块里，这一块只剩数字，模型分不出口径」。

产物：
    data/chunks.jsonl   一行一块
用法：
    python scripts/03_chunk.py
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_JSONL = os.path.join(ROOT, "data", "pages.jsonl")
OUT_JSONL = os.path.join(ROOT, "data", "chunks.jsonl")

CHUNK_CHARS = 800          # 每块目标字数（课件口径）
OVERLAP_CHARS = 120        # 相邻块重叠（课件口径）
TABLE_ROWS_PER_CHUNK = 12  # 一张表按多少行切开
SENT_END = "。！？；\n"      # 往左找这些字符收尾

# 小标题：「一、xxx」「（二）xxx」——比「第X节」细一层，展示出处时更好认
def lines_of(c):
    """块文本里有信息量的行（去掉过短/过长的行）。"""
    return [l.strip() for l in c["raw"].splitlines() if 2 < len(l.strip()) < 80]


SUB_RE = re.compile(r"(?:^|\n) *([一二三四五六七八九十]+、|（[一二三四五六七八九十]+）)[^\n]{0,30}")


def split_text(text, size=CHUNK_CHARS, overlap=OVERLAP_CHARS):
    """定长切 + 句末收尾 + 重叠。返回 [(start, piece), ...]。"""
    pieces, start = [], 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:                                    # 尽量在句末收尾
            cut = max((text.rfind(ch, start + size // 2, end) for ch in SENT_END), default=-1)
            if cut > 0:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append((start, piece))
        if end >= n:
            break
        start = max(end - overlap, start + 1)          # 下一块从重叠处开始
    return pieces


def split_table(table_text, rows_per=TABLE_ROWS_PER_CHUNK):
    """表格：按行切，但每块都重复表头（前 2 行），保证「切碎了还是表」。"""
    rows = [r for r in table_text.split("\n") if r.strip()]
    if not rows:
        return []
    header = rows[:2]
    body = rows[2:] or rows
    out = []
    for i in range(0, len(body), rows_per):
        part = body[i:i + rows_per]
        lines = header + [f"…（接上表第{i+1}行起）"] if i else header
        out.append("\n".join(list(lines) + part))
    return out


def strip_boilerplate(chunks, share=0.15, min_keep=60):
    """去掉页眉页脚这类「每页都长一样」的样板行。

    为什么必须做这一步（实测踩出来的坑）：财报每一页都重复
    「美的集团股份有限公司 / 2026年上半年度财务报表附注 / (除特别注明外，金额单位为人民币千元) / 146」，
    这些样板文字跟任何问题都「像」——它们会以几乎相同的余弦分数霸占召回前几名，
    把真正的答案块挤下去（第一版检索就是这么翻车的：问货币资金，
    返回的 4 块全是只有页眉的空块，一样的分 0.712）。

    判定：同一家公司里，某一行出现在 15% 以上的正文块中，就当成样板行删掉；
    删完不足 min_keep 字的块整块丢弃（它本来也只剩页眉）。
    """
    import collections
    per = collections.defaultdict(collections.Counter)
    cnt = collections.Counter()
    for c in chunks:
        if c["kind"] != "text":
            continue
        cnt[c["code"]] += 1
        for l in set(lines_of(c)):
            per[c["code"]][l] += 1
    kept, dropped = [], 0
    for c in chunks:
        if c["kind"] != "text":
            kept.append(c)
            continue
        n = max(cnt[c["code"]], 1)
        boiler = {l for l, k in per[c["code"]].items() if k / n >= share}
        raw = "\n".join(l for l in lines_of(c) if l not in boiler).strip()
        if len(raw) < min_keep:
            dropped += 1
            continue
        c["raw"] = raw
        c["embed_text"] = f"{c['company']}·{c['section']}\n{raw}"
        c["n_chars"] = len(raw)
        kept.append(c)
    return kept, dropped


def main():
    pages = [json.loads(l) for l in open(IN_JSONL, encoding="utf-8")]
    chunks, cid = [], 0
    for p in pages:
        # --- 正文块 ---
        text = p["text"]
        for k, (off, piece) in enumerate(split_text(text)):
            # 就近取一个小标题（块之前最近的那个「一、xxx」）
            subs = [m.group(0).strip() for m in SUB_RE.finditer(text[:off + len(piece)])]
            sub = subs[-1] if subs else ""
            cid += 1
            chunks.append({
                "id": f"{p['code']}_p{p['page']}_{cid:05d}",
                "code": p["code"], "company": p["company"],
                "section": p["section"], "sub": sub, "page": p["page"],
                "chunk_no": k + 1, "kind": "text",
                # 被编码进向量的文本：归属 + 原文
                "embed_text": f"{p['company']}·{p['section']}\n{piece}",
                "raw": piece, "n_chars": len(piece),
            })
        # --- 表格块 ---
        if p["table"]:
            for k, piece in enumerate(split_table(p["table"])):
                cid += 1
                chunks.append({
                    "id": f"{p['code']}_p{p['page']}_t{cid:05d}",
                    "code": p["code"], "company": p["company"],
                    "section": p["section"], "sub": "表格", "page": p["page"],
                    "chunk_no": k + 1, "kind": "table",
                    "embed_text": f"{p['company']}·{p['section']}·表格\n{piece}",
                    "raw": piece, "n_chars": len(piece),
                })

    chunks, n_dropped = strip_boilerplate(chunks)
    print(f"样板行过滤：丢弃 {n_dropped} 个只剩页眉页脚的块")

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    n_text = sum(1 for c in chunks if c["kind"] == "text")
    n_tbl = len(chunks) - n_text
    chars = sum(c["n_chars"] for c in chunks)
    print(f"共 {len(chunks)} 块（正文 {n_text} + 表格 {n_tbl}），{chars/1e4:.1f} 万字")
    print(f"平均每块 {chars/len(chunks):.0f} 字 -> {OUT_JSONL}")
    for c in chunks[:2]:
        print("\n--- 样例 ---")
        print(f"id={c['id']}  {c['company']} {c['section']} p{c['page']} 第{c['chunk_no']}块")
        print(c["embed_text"][:260])


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                              # noqa: BLE001
            pass
    main()
