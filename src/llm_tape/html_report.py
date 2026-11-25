"""Self-contained, XSS-safe interactive HTML trace viewer for llm-tape cassettes."""

from __future__ import annotations

import html
from collections.abc import Sequence

from llm_tape.cassette import Cassette

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


def render_html_report(cassettes: Sequence[Cassette], *, title: str = "llm-tape Trace") -> str:
    """Render one or more cassettes into a self-contained HTML report."""
    total_calls = sum(len(c.interactions) for c in cassettes)
    total_tokens = sum(i.llm.usage.total_tokens for c in cassettes for i in c.interactions)
    total_cached = sum(i.llm.usage.cached_input_tokens for c in cassettes for i in c.interactions)
    total_reasoning = sum(i.llm.usage.reasoning_tokens for c in cassettes for i in c.interactions)
    total_cost = round(sum(i.llm.estimated_cost_usd for c in cassettes for i in c.interactions), 8)
    total_latency = round(sum(i.latency_ms for c in cassettes for i in c.interactions), 1)

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
        "</div></body></html>\n",
    ]
    return "\n".join(parts)
