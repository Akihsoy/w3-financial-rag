# -*- coding: utf-8 -*-
r"""
第 1 步：下载半年报 PDF
=======================
从巨潮资讯网（cninfo.com.cn）批量下载一组上市公司指定报告期的财报全文。

为什么用巨潮而不是交易所官网：
  上交所 sse.com.cn / 深交所 szse.cn 的披露列表接口对翻页和频次都有限制；
  巨潮是证监会指定的信息披露平台，一份接口同时覆盖沪深两市（含科创板、创业板），
  列表接口 + 静态文件地址都很规整，最适合写脚本。

用法：
    python scripts/01_download.py                # 下载 COMPANY_LIST 里全部公司
    python scripts/01_download.py 000333 000651  # 只下指定的股票代码

产物：
    data/pdf/<代码>_<简称>_<报告期>.pdf
    data/download_log.csv   每份报告：代码/简称/标题/公告日期/URL/字节数
"""
import csv
import json
import os
import sys
import time
import gzip
import urllib.parse
import urllib.request

# ============ 配置：要下载哪些公司、哪一期 ============
COMPANY_LIST = [
    ("000333", "美的集团"), ("000651", "格力电器"), ("600690", "海尔智家"),
    ("000921", "海信家电"), ("002668", "TCL智家"), ("000521", "长虹美菱"),
    ("002032", "苏泊尔"), ("002242", "九阳股份"), ("002508", "老板电器"),
    ("002035", "华帝股份"), ("002959", "小熊电器"), ("002705", "新宝股份"),
    ("688169", "石头科技"), ("603486", "科沃斯"), ("603868", "飞科电器"),
]
# 报告期：半年报用 category_bndbg_szsh，年报用 category_ndbg_szsh
CATEGORY = "category_bndbg_szsh"
# 披露窗口：2026 年半年报在 2026-07 ~ 2026-09 之间披露
SE_DATE = "2026-07-01~2026-10-16"
REPORT_TAG = "2026半年报"          # 只用于文件命名

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(ROOT, "data", "pdf")
LOG_CSV = os.path.join(ROOT, "data", "download_log.csv")

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
STOCK_LIST_URL = "http://www.cninfo.com.cn/new/data/szse_stock.json"


def http_post(url, data, retries=4, timeout=30):
    """带重试的 POST（巨潮接口偶发 504，必须重试）。"""
    body = urllib.parse.urlencode(data).encode()
    headers = {
        "User-Agent": UA,
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Referer": "http://www.cninfo.com.cn/",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
    }
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return json.loads(raw.decode("utf-8"))
        except Exception as e:                        # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"POST {url} 失败: {last}")


def http_download(url, path, retries=4, timeout=120):
    """下载二进制文件，返回字节数；已存在则跳过。"""
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        return os.path.getsize(path)               # 已经下过，跳过（断点续跑）
    headers = {"User-Agent": UA, "Referer": "http://www.cninfo.com.cn/"}
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
            return len(data)
        except Exception as e:                        # noqa: BLE001
            if i == retries - 1:
                raise
            print(f"    重试 {i+1}: {e}")
            time.sleep(2 * (i + 1))
    return 0


def load_org_ids():
    """巨潮的股票列表接口：代码 -> orgId（查询公告必须带上 orgId）。"""
    req = urllib.request.Request(STOCK_LIST_URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    return {item["code"]: item["orgId"] for item in data["stockList"]}


def pick_main_report(anns):
    """一份半年报会同时挂「摘要」「正文」「英文版」「更正公告」，只留正文。"""
    bad = ("摘要", "英文", "更正", "取消", "补充", "问询", "意见", "公告", "说明", "摘要（")
    good = []
    for a in anns:
        title = a["announcementTitle"]
        if "半年度报告" not in title.replace("半年报", "半年度报告"):
            if "半年度报告" not in title:
                continue
        if any(b in title for b in bad):
            continue
        if title.endswith("（英文版）") or "（已取消）" in title:
            continue
        good.append(a)
    # 公告日期新的优先（更正版往往更靠后，但更正已在上面被过滤；这里取最早的主报告）
    good.sort(key=lambda a: a["announcementTime"])
    return good[0] if good else None


def main():
    os.makedirs(PDF_DIR, exist_ok=True)
    wanted = sys.argv[1:] or [c for c, _ in COMPANY_LIST]
    name_of = dict(COMPANY_LIST)

    print("拉取股票代码表（orgId）……")
    org_ids = load_org_ids()

    rows, miss = [], []
    for code in wanted:
        name = name_of.get(code, "")
        org = org_ids.get(code)
        if not org:
            print(f"[跳过] {code} {name}：代码表里没有 orgId")
            miss.append((code, name, "无 orgId"))
            continue
        data = http_post(QUERY_URL, {
            "pageNum": 1, "pageSize": 30, "column": "szse", "tabName": "fulltext",
            "plate": "", "stock": f"{code},{org}", "searchkey": "",
            "category": CATEGORY, "trade": "", "seDate": SE_DATE,
            "sortName": "", "sortType": "", "isHLtitle": "true",
        })
        anns = data.get("announcements") or []
        main_ann = pick_main_report(anns)
        if not main_ann:
            print(f"[未找到] {code} {name}：窗口内没有{REPORT_TAG}正文（共 {len(anns)} 条公告）")
            miss.append((code, name, f"窗口内 {len(anns)} 条公告无正文"))
            continue

        title = main_ann["announcementTitle"]
        url = "http://static.cninfo.com.cn/" + main_ann["adjunctUrl"]
        date = time.strftime("%Y%m%d", time.localtime(main_ann["announcementTime"] / 1000))
        fname = f"{code}_{name}_{REPORT_TAG}.pdf"
        path = os.path.join(PDF_DIR, fname)
        size = http_download(url, path)
        print(f"[OK] {code} {name}  {size/1e6:5.2f} MB  {date}  {title}")
        rows.append({"code": code, "name": name, "title": title, "date": date,
                     "url": url, "bytes": size, "file": fname})
        time.sleep(0.8)                                # 礼貌限速

    with open(LOG_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["code", "name", "title", "date", "url", "bytes", "file"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n下载 {len(rows)} 份，写入 {LOG_CSV}")
    if miss:
        print("未拿到：", miss)


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):                # 中文 Windows 控制台默认 GBK，防崩
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                              # noqa: BLE001
            pass
    main()
