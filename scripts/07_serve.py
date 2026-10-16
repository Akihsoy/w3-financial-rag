# -*- coding: utf-8 -*-
r"""
第 7 步：一条命令起服务，浏览器里问财报
=======================================
只依赖 Python 标准库（http.server），不装 Flask 也不装前端框架——
建库要的东西都在别的脚本里，这里只负责把「提问 → 召回 → 带出处回答」搬上网页。

页面能看到的四样东西：
    ① 提问框 + 检索模式开关（向量 / BM25 / 混合 / 改写+混合）——就是课件实验一到三的对照；
    ② 召回明细：这次命中的每一块（公司 · 章节 · 页码 · 类型 · 向量相似度 · BM25 分）；
    ③ 带 [编号] 出处的回答，点编号跳到对应的块；
    ④ 文本块星空图：全部块降维到二维，看得到「一座小岛一家公司」。

启动：
    python scripts/07_serve.py            # 默认 8765 端口
    python scripts/07_serve.py --port 9000
浏览器打开 http://127.0.0.1:8765
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import retrieval                                   # noqa: E402

STORE = None
REWRITER = None
LOCK = threading.Lock()                            # 模型不是线程安全的，串行处理提问


def get_store():
    global STORE
    if STORE is None:
        t0 = time.time()
        STORE = retrieval.Store()
        print(f"索引加载完成 {time.time()-t0:.1f}s：{len(STORE.chunks)} 块")
    return STORE


class Handler(BaseHTTPRequestHandler):
    server_version = "FinRAG/1.0"

    def log_message(self, fmt, *args):             # 安静点，只打关键行
        pass

    # ---------- 工具 ----------
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False)
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

    # ---------- 路由 ----------
    def do_GET(self):
        store = get_store()
        path = self.path.split("?", 1)[0]          # 路由只看路径，查询串（?q=…&auto=1）留给前端
        if path in ("/", "/index.html"):
            with open(os.path.join(ROOT, "web", "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/api/meta":
            import collections
            co = collections.Counter(c["company"] for c in store.chunks)
            self._send(200, {
                "n_chunks": len(store.chunks),
                "n_text": sum(1 for c in store.chunks if c["kind"] == "text"),
                "n_table": sum(1 for c in store.chunks if c["kind"] == "table"),
                "companies": sorted(co.items(), key=lambda x: -x[1]),
                "model": retrieval.EMBED_MODEL_NAME,
                "answer_model": retrieval.DEEPSEEK_MODEL,
                "has_key": bool(retrieval.DEEPSEEK_KEY),
            })
        elif path.startswith("/api/map"):
            p = os.path.join(ROOT, "data", "map.json")
            if os.path.exists(p):
                self._send(200, open(p, encoding="utf-8").read())
            else:
                self._send(200, {"points": []})
        elif path.startswith("/api/chunk"):
            q = self.path.split("?", 1)[1] if "?" in self.path else ""
            cid = dict(x.split("=", 1) for x in q.split("&") if "=" in x).get("id", "")
            cid = cid.replace("%5F", "_")
            i = store.id2i.get(cid)
            self._send(200, store.chunks[i] if i is not None else {"error": "not found"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        global REWRITER
        if self.path != "/api/ask":
            return self._send(404, {"error": "not found"})
        try:
            req = self._read_json()
        except Exception as e:                     # noqa: BLE001
            return self._send(400, {"error": f"bad json: {e}"})
        q = (req.get("question") or "").strip()
        mode = req.get("mode") or "hybrid"
        top_k = int(req.get("top_k") or 8)
        if not q:
            return self._send(400, {"error": "问题为空"})
        if mode == "rewrite" and REWRITER is None:
            REWRITER = retrieval.make_rewriter()
        store = get_store()
        t0 = time.time()
        with LOCK:
            try:
                hits = store.search(q, mode=mode, top_k=top_k, rewriter=REWRITER)
                from retrieval import build_answer_prompt, ANSWER_SYSTEM, llm_chat
                ans, usage = llm_chat([
                    {"role": "system", "content": ANSWER_SYSTEM},
                    {"role": "user", "content": build_answer_prompt(q, hits)}])
            except Exception as e:                 # noqa: BLE001
                return self._send(500, {"error": str(e)[:500]})
        self._send(200, {
            "question": q, "mode": mode, "seconds": round(time.time() - t0, 1),
            "answer": ans, "usage": usage,
            "hits": [{"rank": h["rank"], "score": h["score"], "sim": h["sim"], "bm25": h["bm25"],
                      "id": h["chunk"]["id"], "company": h["chunk"]["company"],
                      "section": h["chunk"]["section"], "page": h["chunk"]["page"],
                      "kind": h["chunk"]["kind"], "text": h["chunk"]["raw"]} for h in hits],
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    get_store()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"财报问答库已启动： http://127.0.0.1:{args.port}  (Ctrl+C 停止)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                          # noqa: BLE001
            pass
    main()
