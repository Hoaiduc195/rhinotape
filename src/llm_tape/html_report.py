"""Self-contained, XSS-safe interactive HTML trace viewer for llm-tape cassettes."""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from typing import Any

from llm_tape.cassette import Cassette
from llm_tape.models import Interaction

_BASE_CSS = """
:root { color-scheme: light dark; --bg: #0f172a; --card: #1e293b; --fg: #e2e8f0;
  --muted: #94a3b8; --border: #334155; --accent: #38bdf8; --ok: #22c55e; --err: #ef4444; }
@media (prefers-color-scheme: light) {
  :root { --bg: #f8fafc; --card: #ffffff; --fg: #0f172a; --muted: #475569; --border: #cbd5e1;
    --accent: #0284c7; --ok: #16a34a; --err: #dc2626; }
}
body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--fg);
  margin: 0; padding: 24px; line-height: 1.5; }
.wrap { max-width: 1120px; margin: 0 auto; }
h1 { margin: 0 0 16px; font-size: 1.5rem; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px; margin-bottom: 20px; }
.kpi { background: var(--card); border: 1px solid var(--border); border-radius: 8px;
  padding: 12px 16px; }
.kpi .label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; }
.kpi .val { font-size: 1.35rem; font-weight: 700; margin-top: 4px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 8px;
  margin-bottom: 12px; padding: 12px 16px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.75rem;
  border: 1px solid var(--border); margin-right: 6px; }
.ok { color: var(--ok); border-color: var(--ok); }
.err { color: var(--err); border-color: var(--err); }
pre { background: rgba(0,0,0,0.2); padding: 10px; border-radius: 6px; overflow-x: auto;
  font-size: 0.82rem; white-space: pre-wrap; word-break: break-word; }
.bar-wrap { height: 6px; background: var(--border); border-radius: 3px; margin: 8px 0; }
.bar { height: 100%; background: var(--accent); border-radius: 3px; }
.controls { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 16px; }
.controls input, .controls select { background: var(--card); color: var(--fg);
  border: 1px solid var(--border); border-radius: 6px; padding: 6px 10px; font-size: 0.9rem; }
"""


_FILTER_JS = """
(function() {
  const q = document.getElementById('tape-search');
  const prov = document.getElementById('tape-provider');
  const st = document.getElementById('tape-status');
  function apply() {
    const term = (q.value || '').toLowerCase();
    const p = prov.value;
    const s = st.value;
    document.querySelectorAll('.card').forEach(el => {
      const okQ = !term || (el.dataset.search || '').includes(term);
      const okP = !p || el.dataset.provider === p;
      const code = parseInt(el.dataset.status || '0', 10);
      const okS = !s || (s === 'ok' ? code < 400 : code >= 400);
      el.style.display = (okQ && okP && okS) ? '' : 'none';
    });
  }
  q.addEventListener('input', apply);
  prov.addEventListener('change', apply);
  st.addEventListener('change', apply);
})();
"""


def _fmt_body(body: Any) -> str:
    if body is None:
        return ""
    if isinstance(body, (dict, list)):
        return json.dumps(body, indent=2, ensure_ascii=False)
    return str(body)


def _render_interaction_card(
    cassette_name: str, idx: int, inter: Interaction, max_latency: float
) -> str:
    status = inter.response.status_code
    status_cls = "ok" if status < 400 else "err"
    pct = min(100.0, (inter.latency_ms / max_latency) * 100.0) if max_latency > 0 else 0.0
    provider = html.escape(inter.llm.provider)
    model = html.escape(inter.llm.model or "-")
    url = html.escape(inter.request.url)
    method = html.escape(inter.request.method)
    stream_badge = "<span class='badge'>SSE</span>" if inter.llm.is_stream else ""
    tools_str = html.escape(", ".join(tc.name for tc in inter.llm.tool_calls))

    details: list[str] = []
    if inter.llm.thinking:
        details.append(f"<strong>Thinking</strong><pre>{html.escape(inter.llm.thinking)}</pre>")
    if inter.llm.text:
        details.append(f"<strong>Response Text</strong><pre>{html.escape(inter.llm.text)}</pre>")
    if inter.llm.tool_calls:
        tc_json = json.dumps([tc.to_dict() for tc in inter.llm.tool_calls], indent=2)
        details.append(f"<strong>Tool Calls</strong><pre>{html.escape(tc_json)}</pre>")
    req_str = _fmt_body(inter.request.body)
    if req_str:
        details.append(f"<strong>Request Body</strong><pre>{html.escape(req_str)}</pre>")
    if inter.response.stream_chunks is not None:
        details.append(
            f"<strong>Stream ({len(inter.response.stream_chunks)} chunks)</strong>"
            f"<pre>{html.escape(''.join(inter.response.stream_chunks))}</pre>"
        )
    else:
        resp_str = _fmt_body(inter.response.body)
        if resp_str:
            details.append(f"<strong>Response Body</strong><pre>{html.escape(resp_str)}</pre>")

    search_blob = html.escape(
        f"{cassette_name} {inter.llm.provider} {inter.llm.model or ''} "
        f"{inter.request.url} {tools_str} {inter.llm.text}".lower(),
        quote=True,
    )
    return (
        f"<div class='card' data-provider='{provider}' data-status='{status}' "
        f"data-search='{search_blob}'>"
        f"<details><summary>"
        f"<span class='badge {status_cls}'>{status}</span>"
        f"<span class='badge'>{provider}</span>"
        f"<span class='badge'>{model}</span>{stream_badge}"
        f"<strong>[{idx}] {method} {url}</strong> "
        f"<span style='color:var(--muted)'>({inter.latency_ms:.1f}ms | "
        f"{inter.llm.usage.total_tokens} tok | ${inter.llm.estimated_cost_usd:.6f})</span>"
        f"</summary>"
        f"<div class='bar-wrap'><div class='bar' style='width:{pct:.1f}%'></div></div>"
        + "".join(details)
        + "</details></div>"
    )


def render_html_report(cassettes: Sequence[Cassette], *, title: str = "llm-tape Trace") -> str:
    """Render one or more cassettes into a self-contained HTML report."""
    all_interactions = [
        (c.path.name, idx, inter)
        for c in cassettes
        for idx, inter in enumerate(c.interactions, start=1)
    ]
    total_calls = len(all_interactions)
    total_tokens = sum(inter.llm.usage.total_tokens for _, _, inter in all_interactions)
    total_cached = sum(inter.llm.usage.cached_input_tokens for _, _, inter in all_interactions)
    total_reasoning = sum(inter.llm.usage.reasoning_tokens for _, _, inter in all_interactions)
    total_cost = round(sum(inter.llm.estimated_cost_usd for _, _, inter in all_interactions), 8)
    total_latency = round(sum(inter.latency_ms for _, _, inter in all_interactions), 1)
    max_latency = max((inter.latency_ms for _, _, inter in all_interactions), default=0.0)

    providers = sorted({inter.llm.provider for _, _, inter in all_interactions})
    prov_opts = "".join(
        f"<option value='{html.escape(p, quote=True)}'>{html.escape(p)}</option>" for p in providers
    )
    controls_html = (
        "<div class='controls'>"
        "<input id='tape-search' type='search' placeholder='Filter by model, tool, URL, text...'>"
        f"<select id='tape-provider'><option value=''>All providers</option>{prov_opts}</select>"
        "<select id='tape-status'><option value=''>All statuses</option>"
        "<option value='ok'>2xx/3xx OK</option><option value='err'>4xx/5xx Errors</option></select>"
        "</div>"
    )

    cards_html = "\n".join(
        _render_interaction_card(cname, idx, inter, max_latency)
        for cname, idx, inter in all_interactions
    )

    esc_title = html.escape(title)
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{esc_title}</title>",
        f"<style>{_BASE_CSS}</style>",
        "</head><body><div class='wrap'>",
        f"<h1>{esc_title}</h1>",
        "<div class='kpis'>",
        f"<div class='kpi'><div class='label'>Cassettes</div>"
        f"<div class='val'>{len(cassettes)}</div></div>",
        f"<div class='kpi'><div class='label'>Calls</div>"
        f"<div class='val'>{total_calls}</div></div>",
        f"<div class='kpi'><div class='label'>Tokens</div>"
        f"<div class='val'>{total_tokens:,}</div></div>",
        f"<div class='kpi'><div class='label'>Cached / Reasoning</div>"
        f"<div class='val'>{total_cached:,} / {total_reasoning:,}</div></div>",
        f"<div class='kpi'><div class='label'>Est. Cost</div>"
        f"<div class='val'>${total_cost:.6f}</div></div>",
        f"<div class='kpi'><div class='label'>Total Latency</div>"
        f"<div class='val'>{total_latency:.1f}ms</div></div>",
        "</div>",
        controls_html,
        cards_html or "<p>No interactions recorded.</p>",
        f"</div><script>{_FILTER_JS}</script></body></html>\n",
    ]
    return "\n".join(parts)
