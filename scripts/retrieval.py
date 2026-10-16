# -*- coding: utf-8 -*-
r"""
检索与回答的公共模块（被 06_ask.py 和 07_serve.py 共用）
=========================================================
一条完整的问答链，四步：

    问题 ──①向量检索（Qwen3-Embedding + 余弦）──┐
         └─②BM25 检索（jieba 分词 + 倒排）    ──┴─③RRF 合并─→ ④DeepSeek 带出处作答

为什么两条路都要（课件的做法）：向量管「意思相近」，BM25 管「字面必须出现」
——财报里的专有名词、科目名、年份正好是 BM25 的强项，而口语化提问是向量的强项。

四种检索模式（对应课件实验一到实验三的对照）：
    vector 只用向量 / bm25 只用 BM25 / hybrid 两条合并 / rewrite 先用 LLM 把问题
    压成一句检索式再 hybrid。
"""
import collections
import json
import math
import os
import re
import sys
import time
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# ---------- .env：密钥只放这里，且 .env 在 .gitignore 里 ----------
def load_env(path=None):
    path = path or os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

load_env()
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
QUERY_INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"


# ============================ 索引加载 ============================
class Store:
    def __init__(self, load_embedding=True):
        self.chunks = [json.loads(l) for l in open(os.path.join(DATA, "chunks.jsonl"), encoding="utf-8")]
        self.id2i = {c["id"]: i for i, c in enumerate(self.chunks)}
        self.bm25 = None
        self.tokens = None
        p = os.path.join(DATA, "bm25.pkl")
        if os.path.exists(p):
            import pickle
            with open(p, "rb") as f:
                d = pickle.load(f)
            self.bm25, self.tokens = d["bm25"], d["tokens"]
            assert len(self.tokens) == len(self.chunks), "BM25 索引与 chunks 行数不一致，重建索引"
        self.emb = None
        self.model = None
        if load_embedding:
            ep = os.path.join(DATA, "emb.npy")
            if os.path.exists(ep):
                self.emb = np.load(ep)
                assert self.emb.shape[0] == len(self.chunks), "向量行数与 chunks 不一致，重跑 04"

    # ---------- 向量 ----------
    def _ensure_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(EMBED_MODEL_NAME, device=dev)
            if dev == "cuda":
                self.model = self.model.half()
        return self.model

    def vector_search(self, query, k=30):
        if self.emb is None:
            return []
        m = self._ensure_model()
        q = m.encode([f"Instruct: {QUERY_INSTRUCT}\nQuery:{query}"],
                     normalize_embeddings=True, show_progress_bar=False)[0]
        sims = self.emb @ q.astype(np.float32)
        idx = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in idx]

    # ---------- BM25 ----------
    def bm25_search(self, query, k=30):
        if self.bm25 is None:
            return []
        import jieba
        query = re.sub(r"(?<=\d),(?=\d{3})", "", query)      # 去掉千分位逗号，和建索引用同一套
        toks = [t for t in jieba.lcut(query) if t.strip()
                and not (t.isascii() and not t.isalnum() and not any(ch.isdigit() for ch in t))]
        scores = self.bm25.get_scores(toks)
        idx = np.argsort(-np.asarray(scores))[:k]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]

    # ---------- RRF 合并 ----------
    @staticmethod
    def rrf(lists, k=60):
        """Reciprocal Rank Fusion：两条路各按名次给分 1/(60+rank)，加总。
        只比名次不比分数，省得给两套不同量纲的分数调权重。"""
        agg = {}
        for lst in lists:
            for rank, (i, _s) in enumerate(lst):
                agg.setdefault(i, 0.0)
                agg[i] += 1.0 / (k + rank + 1)
        return sorted(agg.items(), key=lambda x: -x[1])

    def search(self, query, mode="hybrid", top_k=8, cand=30, rewriter=None,
               company_aware=False, keywordizer=None):
        """返回 [{chunk, score, rank, from_, sim, bm25}, ...]（前 top_k）

        company_aware=True 时启用「按公司分组召回」（跨公司题的解法）：
          ① 问题里点名的公司，先各自保底召回若干块——大公司块多，
             全局 top-8 会被它一家占满，另外两家根本进不来；
          ② 没点名公司的全景题，则按公司轮转取块，保证前 8 块覆盖尽可能多的公司。
        keywordizer 给 "kw" 模式用：LLM 抽关键词 -> BM25 那条路改走关键词。
        """
        pool = max(cand, 200) if company_aware else cand
        use_kw = mode == "kw"
        v = self.vector_search(query, pool) if mode in ("vector", "hybrid", "rewrite", "kw") else []
        if mode == "rewrite":
            q_for_bm25 = rewriter(query) if rewriter else query
        elif use_kw:
            q_for_bm25 = keywordizer(query) if keywordizer else query
        else:
            q_for_bm25 = query
        b = self.bm25_search(q_for_bm25, pool) if mode in ("bm25", "hybrid", "rewrite", "kw") else []
        if mode == "vector":
            merged = list(v)
        elif mode == "bm25":
            merged = list(b)
        else:                                      # hybrid / rewrite / kw 都走 RRF 合并
            merged = self.rrf([v, b])
        vs, bs = dict(v), dict(b)

        if company_aware and merged:
            named = self.detect_companies(query)
            merged = self.diversify(merged, top_k, named)
        merged = merged[:top_k]

        out = []
        for rank, (i, score) in enumerate(merged):
            out.append({"chunk": self.chunks[i], "score": round(float(score), 4), "rank": rank + 1,
                        "sim": round(vs.get(i, 0.0), 4), "bm25": round(bs.get(i, 0.0), 2)})
        return out

    # ---------- 跨公司召回 ----------
    ALIASES = {   # 问题里可能怎么称呼这家公司
        "美的集团": ["美的"], "格力电器": ["格力"], "海尔智家": ["海尔"],
        "海信家电": ["海信"], "TCL智家": ["TCL", "Tcl"], "长虹美菱": ["长虹", "美菱"],
        "苏泊尔": ["苏泊尔"], "九阳股份": ["九阳"], "老板电器": ["老板电器", "老板"],
        "华帝股份": ["华帝"], "小熊电器": ["小熊"], "新宝股份": ["新宝"],
        "石头科技": ["石头"], "科沃斯": ["科沃斯"], "飞科电器": ["飞科"],
    }

    def detect_companies(self, query):
        found = []
        for comp, alts in self.ALIASES.items():
            if comp in query or any(a in query for a in alts):
                found.append(comp)
        return found

    def diversify(self, merged, top_k, named=None):
        """按公司分组，保证跨公司题每题都能拿到多家的块。

        两种情形分开处理：
          · 问题点名了公司（「美的、格力、海尔三家的销售费用」）——
            每家保底 top_k/家数 块。否则块最多的那家会把 8 个名额全占满，
            另外两家一块都进不来（这是实测：Q7 只召回了美的和海尔，格力没进来）。
          · 全景题没点名公司（「这 15 家普遍提到哪些风险」）——
            前 60% 名额按公司轮转，保证覆盖多家；剩下按分数补齐。
        """
        named = named or []
        picked, seen = [], set()
        if named:
            per = max(3, -(-top_k // len(named)))   # 向上取整，至少 3 块
            for comp in named:
                cnt = 0
                for i, s in merged:
                    if self.chunks[i]["company"] == comp and i not in seen and cnt < per:
                        picked.append((i, s)); seen.add(i); cnt += 1
        else:
            n_rr = max(1, int(top_k * 0.6))
            groups = collections.OrderedDict()
            for i, s in merged:
                if i not in seen:
                    groups.setdefault(self.chunks[i]["company"], []).append((i, s))
            while len(picked) < n_rr and any(groups.values()):
                for comp in list(groups.keys()):
                    if len(picked) >= n_rr:
                        break
                    if groups[comp]:
                        i, s = groups[comp].pop(0)
                        picked.append((i, s)); seen.add(i)
        for i, s in merged:                            # 剩下的按名次补满
            if len(picked) >= top_k:
                break
            if i not in seen:
                picked.append((i, s)); seen.add(i)
        picked.sort(key=lambda x: -x[1])
        return picked


# ============================ 大模型 ============================
def llm_chat(messages, temperature=0.2, max_tokens=8192, timeout=600, retries=3,
             reasoning=None):
    """调 DeepSeek 的 OpenAI 兼容接口。key 只在服务端读环境变量。

    reasoning: None=模型默认 / "none"=关掉思考模式 / "low"|"high"|"max"。
    max_tokens 必须给足：思考模式下 reasoning token 也算在 max_tokens 里，
    设成 2048 时模型可能把 2048 全用在「想」上，正文返回空字符串（实测踩过）。
    课件里踩过的坑：忘了关思考模式时，输出 token 里 81% 是推理过程，
    同样的活又慢又贵十倍——所以简单任务（改写查询）一律关掉。
    """
    if not DEEPSEEK_KEY:
        raise RuntimeError("没有找到 DEEPSEEK_API_KEY —— 请在 .env 里配置（.env 已被 .gitignore 忽略）")
    payload = {"model": DEEPSEEK_MODEL, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens, "stream": False}
    if reasoning:
        payload["reasoning_effort"] = reasoning
    body = json.dumps(payload).encode()
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(
                f"{DEEPSEEK_BASE}/chat/completions", data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {DEEPSEEK_KEY}"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode("utf-8"))
            return d["choices"][0]["message"]["content"], d.get("usage", {})
        except Exception as e:                        # noqa: BLE001
            last = e
            try:
                detail = e.read().decode("utf-8", "replace")[:300]   # noqa: BLE001
            except Exception:                          # noqa: BLE001
                detail = ""
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"调用模型失败: {last} {detail if 'detail' in dir() else ''}")


ANSWER_SYSTEM = """你是家用电器行业上市公司的财报分析助手。
规则：
1. 只依据「材料」回答，不许用你自己的知识补充任何数字或事实；
2. 每个论断后面必须用 [编号] 标注它来自哪一块材料；
3. 材料里没有的，直接写「材料未提供」，不要猜、不要编；
4. 涉及数字时照抄原数（含单位），需要计算时写清算式；
5. 如果不同公司的口径不同（合并/母公司、营业收入/营业总收入），明确指出口径。
回答用中文，先给结论，再给依据。"""


def build_answer_prompt(question, hits):
    lines = []
    for h in hits:
        c = h["chunk"]
        lines.append(f"[{h['rank']}] {c['company']} {c['section']} 第{c['page']}页"
                     f"（{c['kind']}{'，表' if c['kind']=='table' else ''}）\n{c['raw']}")
    return "材料：\n\n" + "\n\n".join(lines) + f"\n\n问题：{question}"


def make_keyword_extractor():
    """课件实验二/三的「先抽关键词再 BM25」：让便宜的模型把问题翻成检索词。

    为什么有用（实测）：问「美的、格力、海尔三家的销售费用分别是多少」，
    原句 BM25 把「分别是」「多少」「三家公司」这些词也算进分数，
    标准答案块（海尔半年报第 33 页的销售费用表）排在第 1628 名；
    换成抽出来的关键词去检索，同一块排到第 22 名。
    坑（课件原文）：抽出来的词本身会带偏——所以这里明确要求只抽
    公司名、科目名、年份这类必须字面出现的词。
    """
    def extract(q):
        try:
            text, _ = llm_chat([
                {"role": "system", "content":
                 "从用户问题里抽出用于『字面检索』的关键词，只输出一行、空格分隔。"
                 "只保留：公司名/简称、财务科目名、业务名词、年份、数字口径（如 合并/母公司）。"
                 "不要疑问词、不要动词、不要解释。"},
                {"role": "user", "content": q}], temperature=0.0, max_tokens=80,
                reasoning="none")
            return text.strip().split("\n")[0][:150] or q
        except Exception:                          # noqa: BLE001
            return q
    return extract


def make_rewriter():
    """把长问题压成一句检索式（课件『问题先洗干净』）。失败就退回原问题。"""
    def rewrite(q):
        try:
            text, _ = llm_chat([
                {"role": "system", "content": "把用户的问题压缩成一句用于检索的短查询，"
                                              "保留公司名、科目名、年份、口径等限定词，只输出这一句，不要解释。"},
                {"role": "user", "content": q}], temperature=0.0, max_tokens=120,
                reasoning="none")
            return text.strip().split("\n")[0][:120] or q
        except Exception:                              # noqa: BLE001
            return q
    return rewrite


def answer(store, question, mode="hybrid", top_k=8, rewriter=None):
    hit = store.search(question, mode=mode, top_k=top_k, rewriter=rewriter)
    if not hit:
        return {"answer": "没有检索到相关材料。", "hits": [], "usage": {}}
    text, usage = llm_chat([
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": build_answer_prompt(question, hit)}],
        reasoning=os.environ.get("DEEPSEEK_REASONING_ANSWER", "high"))
    return {"answer": text, "hits": hit, "usage": usage}
