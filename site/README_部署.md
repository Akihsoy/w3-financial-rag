# 部署到 Vercel（静态演示站）

这个目录是一套**纯静态**页面：没有后端、没有构建步骤、不含任何密钥。
所有内容（10 道题的回答、召回明细、排名、判定、星空图）都是本地真实运行结果的固化。

## 部署（30 秒，二选一）

**方式 A：把 site 设为根目录（推荐）**
1. Vercel → Add New → Project → 选这个仓库
2. **Root Directory 选 `site`**（其余全默认：Framework Preset = Other，不需要 Build Command）
3. Deploy → 打开域名，看到的就是本站

**方式 B：直接部署仓库根目录**
仓库根目录有一个 `index.html`，会自动跳到 `site/index.html`，功能一样，只是地址里多一段 `/site/`。

## 本地预览

```bash
cd site
python -m http.server 8080      # 打开 http://127.0.0.1:8080
```
⚠️ 直接双击 `index.html`（file://）不行——页面用 `fetch` 读 `demo.json`/`map.json`，浏览器会拦。

## 页面数据怎么更新

```bash
python scripts/11_build_site.py     # 重新生成 site/demo.json 与 site/map.json
```
改完 `results/` 里的实验记录（比如重新跑题、改判卷），重跑这条命令即可。

## 想做成「真·在线提问」要加什么

现在这套问答依赖三样静态托管给不了的东西：**1.2 GB 的向量模型**（torch + Qwen3-Embedding）、
**numpy 索引与 jieba 倒排**、**带密钥的 DeepSeek 调用**。要在 Vercel 上真跑起来，至少需要：

1. **嵌入换成 API**：把本机模型换成商业嵌入接口（阿里百炼 text-embedding-v4 / OpenAI 等），
   查询时只调用一次 API，不需要在函数里加载模型；
   → 但**建库时的 5876 个向量仍要在本地算好**，把 `emb.npy` 作为静态资源上传（约 24 MB）。
2. **检索逻辑移植到函数**：Vercel Python Function 里做「余弦 + BM25」——BM25 需要把
   倒排索引（`bm25.pkl`，含分词结果）也一起打包，注意函数体积上限（未压缩 250 MB）。
   更省事的做法是**只保留向量检索**，把 BM25 那份去掉。
3. **密钥托管**：`DEEPSEEK_API_KEY` 放进 Vercel 的环境变量（只服务端可见），
   **绝不能写进前端**。同时想清楚：公开站点=任何人都在花你的额度，
   建议加访问口令或限流。

结论：静态版适合当作品集展示；真要在线问答，成本最低的其实是找一台能跑 Python 的
小机器（或自己的笔记本）跑 `scripts/07_serve.py`，用内网穿透暴露出去。
