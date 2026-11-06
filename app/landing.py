# -*- coding: utf-8 -*-
"""首页：一个能直接用的问答界面（无构建步骤，单文件 HTML）"""

INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hybrid RAG Agent · 文档问答</title>
<style>
  :root{--fg:#1a1a1a;--muted:#6b6b6b;--line:#e6e4df;--accent:#1f3b63;--bg:#faf9f6}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:15px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}
  .wrap{max-width:720px;margin:0 auto;padding:56px 20px 80px}
  h1{font-size:27px;margin:0 0 8px;letter-spacing:-.02em;font-weight:650}
  .sub{color:var(--muted);font-size:14px;margin-bottom:28px}
  .card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px;
        box-shadow:0 1px 2px rgba(0,0,0,.03)}
  textarea{width:100%;border:1px solid var(--line);border-radius:8px;padding:12px 14px;
           font:inherit;color:inherit;resize:vertical;min-height:76px;background:#fdfdfc}
  textarea:focus{outline:none;border-color:var(--accent);background:#fff}
  .row{display:flex;align-items:center;gap:12px;margin-top:10px}
  button{background:var(--accent);color:#fff;border:0;border-radius:8px;
         padding:9px 22px;font:inherit;cursor:pointer;transition:opacity .15s}
  button:hover:not(:disabled){opacity:.88}
  button:disabled{opacity:.45;cursor:default}
  .hint{color:var(--muted);font-size:12.5px}
  #out{margin-top:22px;display:none}
  .answer{white-space:pre-wrap;word-break:break-word}
  .meta{margin-top:16px;padding-top:14px;border-top:1px solid var(--line);
        font-size:12.5px;color:var(--muted);display:flex;flex-wrap:wrap;gap:16px}
  .meta b{font-weight:600;color:#444}
  .err{color:#b3261e}
  footer{margin-top:36px;font-size:12.5px;color:var(--muted);line-height:1.9}
  a{color:var(--accent);text-decoration:none;border-bottom:1px solid #c9d3e2}
  .spin{display:inline-block;width:13px;height:13px;border:2px solid #fff;
        border-top-color:transparent;border-radius:50%;animation:s .7s linear infinite;
        vertical-align:-2px;margin-right:7px}
  @keyframes s{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="wrap">
  <h1>Hybrid RAG Agent</h1>
  <div class="sub">混合检索（BM25 + 语义向量 + RRF）· 20 题评测 Hit@3 95% / MRR 0.771 · 知识库：本项目文档（README / 踩坑 / 部署）</div>

  <div class="card">
    <textarea id="q" placeholder="例如：这个项目是怎么部署的？踩过哪些坑？"></textarea>
    <div class="row">
      <button id="go" onclick="ask()">提问</button>
      <span class="hint">首次请求需加载模型，约 3–10 秒</span>
    </div>
    <div id="out">
      <div class="answer" id="ans"></div>
      <div class="meta" id="meta"></div>
    </div>
  </div>

  <footer>
    接口：<a href="/docs">/docs</a> · 健康检查：<a href="/health">/health</a><br>
    源码：<a href="https://github.com/SukiAoi/hybrid-rag-agent" target="_blank" rel="noopener">github.com/SukiAoi/hybrid-rag-agent</a>
  </footer>
</div>

<script>
async function ask() {
  const q = document.getElementById('q').value.trim();
  const btn = document.getElementById('go');
  const out = document.getElementById('out');
  const ans = document.getElementById('ans');
  const meta = document.getElementById('meta');
  if (!q) { document.getElementById('q').focus(); return; }

  btn.disabled = true;
  btn.innerHTML = '<span class="spin"></span>检索中';
  out.style.display = 'block';
  ans.className = 'answer';
  ans.textContent = '';
  meta.innerHTML = '';

  const t0 = performance.now();
  try {
    const r = await fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q})
    });
    const data = await r.json();
    const ms = Math.round(performance.now() - t0);
    if (!r.ok) {
      ans.className = 'answer err';
      ans.textContent = (typeof data.detail === 'string')
        ? data.detail
        : ('请求失败（HTTP ' + r.status + '）');
    } else {
      ans.textContent = data.answer || '（无内容）';
      const src = (data.sources && data.sources.length) ? data.sources.join('、') : '—';
      const tools = (data.tools_used && data.tools_used.length) ? data.tools_used.join(' → ') : '未调用工具';
      meta.innerHTML = '<span><b>来源</b> ' + src + '</span>' +
                       '<span><b>工具</b> ' + tools + '</span>' +
                       '<span><b>耗时</b> ' + ms + ' ms</span>';
    }
  } catch (e) {
    ans.className = 'answer err';
    ans.textContent = '网络错误：' + e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = '提问';
  }
}

document.getElementById('q').addEventListener('keydown', function (e) {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') ask();
});
</script>
</body>
</html>
"""
