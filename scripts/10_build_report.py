# -*- coding: utf-8 -*-
r"""
第 10 步：把实验数据汇总成《10 题逐题记录》
===========================================
把三份数据合到一起，生成 results/逐题记录.md：

    · results/recall_eval.json  四种检索模式各自召回了哪些块、答案块排第几
    · results/qa_hybrid.json    基本混合模式的回答
    · results/qa_kw_group.json  抽词+分组模式的回答
    · results/判卷.json         **人工判卷结果**（对/错/部分对 + 错在哪）——
                                这一项必须人写：判卷标准是「回答是否与半年报原文一致」，
                                不能拿被测系统自己的输出当标准答案

用法：python scripts/10_build_report.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def load(name, default=None):
    p = os.path.join(RES, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else (default if default is not None else {})


def main():
    ev = {r["id"]: r for r in load("recall_eval.json", [])}
    hy = {r["qid"]: r for r in load("qa_hybrid.json", [])}
    kg = {r["qid"]: r for r in load("qa_kw_group.json", [])}
    verdict = load("判卷.json")
    questions = json.load(open(os.path.join(ROOT, "data", "questions.json"), encoding="utf-8"))

    out = ["# 10 道题逐题记录", "",
           "> 语料：15 家家电公司 2026 年半年报（2588 页 / 5876 块）。",
           "> 「标准答案块」是人读原文标的；「答对没有」是拿回答和半年报原文逐条核对得出的。",
           "> 召回列的含义：**首个答案块的排名 / 前 8 块覆盖了几家公司**，`—` 表示前 8 块里没有答案块。", ""]

    modes = ["vector", "bm25", "hybrid", "rewrite", "hybrid+分组", "抽词+分组"]
    summary = []
    for item in questions:
        qid = item["id"]
        e, v = ev.get(qid, {}), verdict.get(str(qid), {})
        out.append("---")
        out.append(f"## Q{qid} · {item['type']}")
        out.append(f"**问题**：{item['q']}")
        out.append("")
        out.append("**召回**（排名 / 覆盖公司数）")
        out.append("")
        out.append("| 检索模式 | " + " | ".join(modes) + " |")
        out.append("|---|" + "---|" * len(modes))
        cells = []
        for m in modes:
            d = (e.get("modes") or {}).get(m)
            if not d:
                cells.append("—")
            else:
                r = d.get("first_gold_rank")
                cells.append(f"{r if r else '—'} / {d.get('n_companies')}家")
        out.append("| | " + " | ".join(cells) + " |")
        out.append("")
        if e.get("gold_ids"):
            out.append(f"标准答案块 {len(e['gold_ids'])} 个（含答案数值的块）："
                       + "、".join(f"`{x}`" for x in e["gold_ids"][:6])
                       + ("…" if len(e["gold_ids"]) > 6 else ""))
            out.append("")

        if v:
            out.append(f"**判定：{v.get('verdict', '')}**")
            if v.get("right"):
                out.append(f"- 答对的部分：{v['right']}")
            if v.get("wrong"):
                out.append(f"- 错/漏的部分：{v['wrong']}")
            if v.get("why"):
                out.append(f"- 原因：{v['why']}")
            out.append("")

        for label, d in (("基本混合 hybrid", hy.get(qid)), ("抽词+分组", kg.get(qid))):
            if not d:
                continue
            out.append(f"<details><summary>{label} · 召回 {len(d['hits'])} 块 · {d['seconds']}s"
                       f" · {d.get('usage', {}).get('total_tokens')} tokens</summary>")
            out.append("")
            out.append("召回块：" + "；".join(
                f"[{h['rank']}] {h['company']} {h['section']} p{h['page']}"
                f"（{'表格' if h['kind'] == 'table' else '正文'}，向量{h['sim']:.2f} / BM25 {h['bm25']:.0f}）"
                for h in d["hits"]))
            out.append("")
            out.append("回答：")
            out.append("")
            out.append(d["answer"] or "（空——思考模式吃光了 max_tokens，见踩坑记录）")
            out.append("")
            out.append("</details>")
            out.append("")
        summary.append((qid, item["type"], v.get("verdict", "未判"), e))

    head = out[:6]
    tail = ["", "---", "", "## 汇总", "", "| 题 | 类型 | 判定 |", "|---|---|---|"]
    for qid, t, verdict_s, _ in summary:
        tail.append(f"| Q{qid} | {t} | {verdict_s} |")

    path = os.path.join(RES, "逐题记录.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(head + tail + [""] + out[6:]))
    print(f"-> {path}")


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                          # noqa: BLE001
            pass
    main()
