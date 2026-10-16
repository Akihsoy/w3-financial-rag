# -*- coding: utf-8 -*-
r"""
第 2 步：PDF -> 带页码标记的纯文本 + 按行列还原的表格
=====================================================
半年报是「正文 + 表格」混合体：答案一半在正文里，一半在三张表和附注里。
所以提取要做两件事：

  ① 正文：按页取文字，去掉 PDF 换行带来的多余空格；
  ② 表格：用 PyMuPDF 的 find_tables() 找表格区域，按行列还原成
     「科目 | 本期 | 上期」这样的行文本；表格区域内的原始文字要剔除，
     否则同一份数据会以「散字」和「表格」两种形态各进一次库。

产物：
    data/pages.jsonl        每页一条：公司/代码/页码/章节/正文/表格
    data/text/<代码>_<简称>.txt   人眼校对用的整份文本（带 [p12] 页码标记）

用法：
    python scripts/02_extract.py            # 处理 data/pdf 下全部 PDF
    python scripts/02_extract.py 000333     # 只处理某一家
"""
import glob
import json
import os
import re
import sys
import time

import pymupdf                                  # noqa: N813  (PyMuPDF >= 1.24 的新名字)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(ROOT, "data", "pdf")
OUT_JSONL = os.path.join(ROOT, "data", "pages.jsonl")
TXT_DIR = os.path.join(ROOT, "data", "text")

# 一级章节：正文里写作「第三节 管理层讨论与分析」
SEC_RE = re.compile(r"第 *([一二三四五六七八九十]+) *节 *([^\n]{0,30})")
# 二级小标题：一、二、；（一）（二）……
SUB_RE = re.compile(r"^\s*(?:[一二三四五六七八九十]+、|（[一二三四五六七八九十]+）)\s*(\S[^\n]{0,30})", re.M)


def squeeze(text: str) -> str:
    """PDF 把「2026 年 8 月」抽成带空格的样子，这里把中文之间的空格挤掉。

    注意：只动空格，**绝不动换行**——换行是段落和表格行的边界，
    早先版本用 \\s+ 连换行一起吃掉，结果整张利润表被串成一行。
    """
    text = text.replace("　", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<=[一-鿿]) +(?=[一-鿿])", "", text)
    text = re.sub(r"(?<=[0-9%]) +(?=[一-鿿])", "", text)
    text = re.sub(r"(?<=[一-鿿]) +(?=[0-9])", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_section(name: str) -> str:
    """章节名只留标题本身：「第八节 财务报告一、审计报告…」-> 「第八节 财务报告」。"""
    name = name.strip()
    name = re.split(r"[（(【”\"'·]|[一二三四五六七八九十]+、|：|:|\.{2,}|…", name)[0]
    return name.strip().rstrip("、，。;；").strip()[:16]


def table_to_lines(table):
    """表格 -> 一行行「单元格 | 单元格 | ……」，空单元格补 '-'。"""
    try:
        rows = table.extract()
    except Exception:                                # noqa: BLE001
        return []
    lines = []
    for row in rows:
        cells = [(c or "").replace("\n", " ").strip() for c in row]
        if not any(cells):
            continue
        lines.append(" | ".join(c or "-" for c in cells))
    return lines


def in_bbox(bbox, boxes, shrink=2.0):
    """文本块的中心点是否落在某个表格框内。"""
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    for b in boxes:
        if b[0] - shrink <= cx <= b[2] + shrink and b[1] - shrink <= cy <= b[3] + shrink:
            return True
    return False


def parse_pdf(path):
    code = os.path.basename(path).split("_")[0]
    name = os.path.basename(path).split("_")[1]
    doc = pymupdf.open(path)
    pages, cur_sec, n_tbl, n_page_tbl = [], "封面与目录", 0, 0
    for i in range(doc.page_count):
        page = doc[i]
        pno = i + 1

        # ---- 表格：先找框，再剔除框内文字，最后按行列还原 ----
        try:
            tabs = page.find_tables()
            table_objs = list(tabs.tables)
        except Exception:                            # noqa: BLE001
            table_objs = []
        boxes = [t.bbox for t in table_objs]
        tbl_lines = []
        for t in table_objs:
            rows = table_to_lines(t)
            if rows:
                n_tbl += 1
                if n_page_tbl < 5 and pno < 12:      # 前几页的表，抽查用
                    pass
                tbl_lines.append(rows)
        n_page_tbl += len(table_objs)

        # ---- 正文：按块取字，落在表格框里的块跳过 ----
        blocks = page.get_text("blocks")
        body_parts = []
        for b in blocks:
            if len(b) >= 7 and b[6] != 0:            # 图片块
                continue
            if in_bbox(b, boxes):
                continue
            body_parts.append(b[4])
        body = squeeze("\n".join(body_parts))

        # ---- 章节跟踪 ----
        m = SEC_RE.search(body)
        if m:
            cur_sec = f"第{m.group(1)}节 {clean_section(m.group(2))}".strip()

        tbl_text = ""
        if tbl_lines:
            flat = []
            for rows in tbl_lines:
                flat.extend(rows)
            tbl_text = squeeze("\n".join(flat))

        pages.append({
            "company": name, "code": code, "page": pno,
            "section": cur_sec, "text": body, "table": tbl_text,
        })
    doc.close()
    return pages, n_tbl


def main():
    os.makedirs(TXT_DIR, exist_ok=True)
    only = sys.argv[1:]
    pdfs = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if only:
        pdfs = [p for p in pdfs if os.path.basename(p).split("_")[0] in only]

    all_pages = []
    seen = set()
    if os.path.exists(OUT_JSONL):                    # 断点续跑
        with open(OUT_JSONL, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                all_pages.append(rec)
                seen.add(rec["code"])

    for path in pdfs:
        code = os.path.basename(path).split("_")[0]
        if code in seen:
            print(f"[跳过] {os.path.basename(path)} 已提取")
            continue
        t0 = time.time()
        pages, n_tbl = parse_pdf(path)
        all_pages.extend(pages)
        chars = sum(len(p["text"]) + len(p["table"]) for p in pages)
        print(f"[OK] {os.path.basename(path):45s} {len(pages):>4}页 "
              f"{chars/1e4:>7.1f}万字 {n_tbl:>5}表 {time.time()-t0:5.1f}s")
        # 人眼校对用
        with open(os.path.join(TXT_DIR, f"{code}_{pages[0]['company']}.txt"), "w", encoding="utf-8") as f:
            for p in pages:
                f.write(f"\n[p{p['page']}] <<{p['section']}>>\n{p['text']}\n")
                if p["table"]:
                    f.write("【表格】\n" + p["table"] + "\n")
        with open(OUT_JSONL, "w", encoding="utf-8") as f:   # 每处理完一家就落盘
            for rec in all_pages:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n合计 {len(all_pages)} 页 -> {OUT_JSONL}")


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                            # noqa: BLE001
            pass
    main()
