"""LifeMate Workbench — Offline Vietnamese Voice-to-Action Agent & V9 Safety Console.

Runs real `src/` Python modules on CPU (semantic routing, number/datetime normalization,
V9 schema-first guardrails, and sandbox trace inspection) inside a clean product interface.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.scenarios import (  # noqa: E402
    L4_PEAK_CHUNKED_VRAM_GIB,
    L4_TOTAL_VRAM_GIB,
    PRIVATE_BENCHMARK_METRICS,
    SCENARIO_CATALOG,
    SCORE_PROGRESSION,
    V6_BEST_VAL_LOSS,
    V7_LOSS_CURVE,
    get_architecture_layers_vi,
    get_v9_evolution_rows,
    run_audit_suite_vi,
    run_workbench_turn,
)

GITHUB_REPO_URL = "https://github.com/Phongkarazz/lifemate-vietnamese-voice-agent"
CASE_STUDY_URL = f"{GITHUB_REPO_URL}/blob/main/docs/ARCHITECTURE_CASE_STUDY.md"

PRESET_CHIPS: list[tuple[str, str]] = [
    ("fintech_bill_fallback", "Trả tiền điện"),
    ("v8_food_menu_dependency", "Đặt phở"),
    ("spoken_id_read_before_write", "Mua siêu thị"),
    ("conditional_weather_guard", "Nhắc thời tiết"),
]

st.set_page_config(
    page_title="LifeMate — Trợ lý tác vụ tiếng Việt ngoại tuyến",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Product Design System (Linear / Stripe / Zalo Daylight) — Zero Streamlit Chrome
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-canvas: #FFFFFF;
    --bg-subtle: #F8F9FB;
    --bg-hover: #F1F5F9;
    --border: #E6E8EC;
    --border-strong: #CBD5E1;
    --ink: #0F172A;
    --ink-2: #475569;
    --ink-3: #64748B;
    --accent: #0068FF;
    --accent-soft: #EFF6FF;
    --ok: #059669;
    --ok-soft: #ECFDF5;
    --warn: #D97706;
    --warn-soft: #FFFBEB;
    --err: #DC2626;
    --err-soft: #FEF2F2;
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background-color: var(--bg-canvas);
    color: var(--ink);
}

code, pre, .mono {
    font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
}

/* Completely remove default Streamlit chrome (Deploy button, toolbar, header, menu, footer) */
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
.stDeployButton,
#MainMenu,
footer,
header[data-testid="stHeader"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}

/* Hide raw unformatted Streamlit slider thumb/tick labels (we render clean VND label above) */
[data-testid="stThumbValue"],
[data-testid="stSliderThumbValue"],
[data-testid="stTickBarMin"],
[data-testid="stTickBarMax"],
[data-testid="stSliderTickBarMin"],
[data-testid="stSliderTickBarMax"] {
    display: none !important;
}

.block-container,
[data-testid="stMainBlockContainer"] {
    max-width: 1120px;
    margin: 0 auto;
    padding: 1.05rem 2rem 2.2rem 2rem !important;
}

/* Minimal 52px Top Bar */
.lm-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding-bottom: 14px;
    margin-bottom: 12px;
    border-bottom: 1px solid var(--border);
}
.lm-brand-wrap {
    display: flex;
    align-items: center;
    gap: 11px;
}
.lm-brand-name {
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--ink);
    display: inline-flex;
    align-items: center;
    gap: 7px;
}
.lm-ver-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.70rem;
    font-weight: 500;
    color: var(--ink-2);
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    padding: 1px 6px;
    border-radius: 999px;
}
.lm-brand-sub {
    font-size: 0.80rem;
    color: var(--ink-3);
    margin-top: 1px;
}
.lm-top-links {
    display: flex;
    align-items: center;
    gap: 14px;
    font-size: 0.82rem;
    white-space: nowrap;
}
.lm-top-links a {
    color: var(--ink-2);
    text-decoration: none;
    font-weight: 500;
    padding: 5px 10px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg-canvas);
    transition: border-color 0.12s ease, color 0.12s ease;
}
.lm-top-links a:hover {
    color: var(--accent);
    border-color: var(--accent);
}

/* Crisp Horizontal Navigation Tabs */
[data-baseweb="tab-list"] {
    gap: 24px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
}
[data-baseweb="tab"] {
    padding: 8px 2px 10px 2px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: var(--ink-2) !important;
    background: transparent !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: var(--accent) !important;
    font-weight: 600 !important;
}

/* Buttons & Composer Form Polish */
div[data-testid="stButton"] button {
    border-radius: 999px;
    border: 1px solid var(--border);
    background-color: var(--bg-canvas);
    color: var(--ink-2);
    font-size: 0.78rem;
    font-weight: 500;
    padding: 0.34rem 0.55rem;
    white-space: nowrap !important;
    min-height: 34px;
    transition: border-color 0.12s ease, background-color 0.12s ease, color 0.12s ease;
}
div[data-testid="stButton"] button:hover {
    border-color: var(--border-strong);
    background-color: var(--bg-subtle);
    color: var(--ink);
}
div[data-testid="stForm"] {
    border: none !important;
    padding: 0 !important;
    margin-top: 10px;
}
div[data-testid="stFormSubmitButton"] button {
    border-radius: 8px !important;
    background-color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    height: 40px;
}
div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #0056D6 !important;
    border-color: #0056D6 !important;
}

/* Left Column: Product Conversation Box */
.lm-chat-box {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--bg-canvas);
    padding: 18px 18px 16px 18px;
    margin-top: 8px;
}
.lm-msg-user-wrap {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 16px;
}
.lm-msg-user {
    max-width: 88%;
    background: var(--accent-soft);
    color: var(--ink);
    border: 1px solid #DBEAFE;
    border-radius: 12px 12px 3px 12px;
    padding: 10px 14px;
    font-size: 0.88rem;
    line-height: 1.5;
}
.lm-msg-bot-head {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.lm-bot-dot {
    width: 18px;
    height: 18px;
    border-radius: 5px;
    background: var(--accent);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: #FFFFFF;
    font-size: 0.65rem;
    font-weight: 700;
}
.lm-bot-name {
    font-size: 0.80rem;
    font-weight: 600;
    color: var(--ink);
}
.lm-bot-mode {
    font-size: 0.72rem;
    color: var(--ink-3);
}
.lm-msg-bot-text {
    font-size: 0.88rem;
    color: var(--ink);
    line-height: 1.55;
    margin-bottom: 12px;
    padding-left: 26px;
}

/* Inline Action Receipt Card */
.lm-receipt {
    margin-left: 26px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg-subtle);
    padding: 11px 14px;
}
.lm-receipt-ok { border-left: 3px solid var(--ok); }
.lm-receipt-warn { border-left: 3px solid var(--warn); }
.lm-receipt-err { border-left: 3px solid var(--err); background: #FFF8F8; }
.lm-rc-badge {
    font-size: 0.72rem;
    font-weight: 600;
    margin-bottom: 3px;
}
.rc-ok { color: var(--ok); }
.rc-warn { color: var(--warn); }
.rc-err { color: var(--err); }
.lm-rc-title {
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--ink);
}
.lm-rc-sub {
    font-size: 0.78rem;
    color: var(--ink-2);
    margin-top: 2px;
}

/* Right Column: Live Execution Inspector */
.lm-insp-panel {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--bg-subtle);
    padding: 11px 14px;
    margin-bottom: 10px;
}
.lm-insp-subhead {
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    color: var(--ink-3);
    margin-bottom: 6px;
}
.lm-kv-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 5px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.82rem;
}
.lm-kv-row:last-child { border-bottom: none; padding-bottom: 1px; }
.lm-kv-k { color: var(--ink-2); }
.lm-kv-r {
    display: flex;
    align-items: center;
    gap: 8px;
    text-align: right;
    font-variant-numeric: tabular-nums;
}
.lm-kv-v { font-weight: 600; color: var(--ink); }
.lm-delta {
    font-size: 0.72rem;
    font-weight: 500;
    padding: 1px 7px;
    border-radius: 4px;
    white-space: nowrap;
}
.d-ok { color: var(--ok); background: var(--ok-soft); }
.d-warn { color: var(--warn); background: var(--warn-soft); }
.d-err { color: var(--err); background: var(--err-soft); }
.d-neutral { color: var(--ink-3); background: #ECEFF4; }

/* Vertical Hairline Execution Rail */
.lm-rail {
    position: relative;
    padding-left: 20px;
    margin: 4px 0 12px 6px;
}
.lm-rail::before {
    content: "";
    position: absolute;
    left: 4px;
    top: 6px;
    bottom: 10px;
    width: 1.5px;
    background: var(--border-strong);
}
.lm-rail-item {
    position: relative;
    padding-bottom: 9px;
}
.lm-rail-item:last-child { padding-bottom: 2px; }
.lm-rail-dot {
    position: absolute;
    left: -20px;
    top: 4px;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    border: 2px solid #FFFFFF;
    box-shadow: 0 0 0 1px var(--border-strong);
}
.dot-ok { background: var(--ok); }
.dot-warn { background: var(--warn); }
.dot-err { background: var(--err); }
.lm-rail-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 8px;
}
.lm-rail-tool {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.80rem;
    font-weight: 600;
    color: var(--ink);
}
.lm-rail-status {
    font-size: 0.73rem;
    font-weight: 600;
    white-space: nowrap;
}
.lm-arg-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 5px;
}
.lm-arg {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--ink-2);
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1px 6px;
    white-space: nowrap;
    max-width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
}
.lm-rail-note {
    font-size: 0.78rem;
    color: var(--ink-2);
    margin-top: 4px;
    line-height: 1.4;
}

/* Custom Hairline Tables (Audit Console & Engineering Benchmarks) */
.lm-table-wrap {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    background: var(--bg-canvas);
    margin: 10px 0 16px 0;
}
table.lm-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.83rem;
}
table.lm-table th {
    background: var(--bg-subtle);
    color: var(--ink-3);
    font-size: 0.73rem;
    font-weight: 600;
    text-align: left;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
}
table.lm-table td {
    padding: 11px 14px;
    border-bottom: 1px solid var(--border);
    color: var(--ink);
    vertical-align: top;
    line-height: 1.45;
}
table.lm-table tr:last-child td { border-bottom: none; }
table.lm-table th.num, table.lm-table td.num {
    text-align: right;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}
table.lm-table th.v9col, table.lm-table td.v9col {
    background: #F5F9FF;
    color: var(--ink);
    font-weight: 600;
}
.lm-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.76rem;
    font-weight: 600;
    color: var(--ok);
    background: var(--ok-soft);
    padding: 2px 8px;
    border-radius: 999px;
    white-space: nowrap;
}
.lm-code-sub {
    display: block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.70rem;
    color: var(--ink-3);
    margin-top: 3px;
}

/* Typographic Editorial Metrics Strip (Tab 3) */
.lm-ed-bar {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    padding: 14px 0;
    margin: 4px 0 18px 0;
}
.lm-ed-item {
    padding: 0 18px;
    border-right: 1px solid var(--border);
}
.lm-ed-item:first-child { padding-left: 2px; }
.lm-ed-item:last-child { border-right: none; }
.lm-ed-k {
    font-size: 0.74rem;
    color: var(--ink-3);
    font-weight: 500;
}
.lm-ed-v {
    font-size: 1.32rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--ink);
    font-variant-numeric: tabular-nums;
    margin: 2px 0;
}
.lm-ed-sub {
    font-size: 0.75rem;
    color: var(--ink-2);
}

.lm-summary-bar {
    font-size: 0.86rem;
    color: var(--ink-2);
    padding-top: 4px;
}
.lm-summary-bar strong { color: var(--ink); }

.lm-sec-head {
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--ink);
    margin: 8px 0 4px 0;
}
.lm-caption {
    font-size: 0.76rem;
    color: var(--ink-3);
    margin-top: -6px;
    margin-bottom: 14px;
}

@media (max-width: 860px) {
    .lm-ed-bar { grid-template-columns: repeat(2, minmax(0, 1fr)); row-gap: 12px; }
    .lm-ed-item:nth-child(2n) { border-right: none; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Rendering Helpers
# ---------------------------------------------------------------------------
def _fmt_vnd(amount: int | float) -> str:
    return f"{int(amount):,}".replace(",", ".") + " đ"


def _render_args_chips(args: dict[str, Any]) -> str:
    if not args:
        return '<div class="lm-arg-wrap"><span class="lm-arg">(không tham số)</span></div>'
    chips = []
    for k, v in args.items():
        val_txt = json.dumps(v, ensure_ascii=False)
        if len(val_txt) > 38:
            val_txt = val_txt[:35] + "..."
        chips.append(f'<span class="lm-arg">{html.escape(str(k))}={html.escape(val_txt)}</span>')
    return f'<div class="lm-arg-wrap">{"".join(chips)}</div>'


def _plotly_editorial(fig: go.Figure, height: int = 285) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#0F172A", family="Inter, sans-serif", size=12),
        title_font=dict(size=13, color="#0F172A"),
        height=height,
        margin=dict(l=30, r=14, t=44, b=30),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E6E8EC", zeroline=False)
    fig.update_yaxes(gridcolor="#F1F5F9", linecolor="#E6E8EC", zeroline=False)
    return fig


@st.cache_data
def _cached_v9_evolution() -> list[dict[str, Any]]:
    return get_v9_evolution_rows()


def _apply_preset_chip(sid: str) -> None:
    st.session_state["workbench_hint"] = sid
    st.session_state["workbench_input"] = SCENARIO_CATALOG[sid]["utterance"]


# ---------------------------------------------------------------------------
# Session Initialization
# ---------------------------------------------------------------------------
if "workbench_hint" not in st.session_state:
    st.session_state["workbench_hint"] = "fintech_bill_fallback"
if "workbench_input" not in st.session_state:
    st.session_state["workbench_input"] = SCENARIO_CATALOG["fintech_bill_fallback"]["utterance"]
if "workbench_guard" not in st.session_state:
    st.session_state["workbench_guard"] = True
if "wb_wallet" not in st.session_state:
    st.session_state["wb_wallet"] = 120_000
if "wb_rain" not in st.session_state:
    st.session_state["wb_rain"] = False


# ---------------------------------------------------------------------------
# Minimalist Product Top Bar (52px)
# ---------------------------------------------------------------------------
st.markdown(
    f"""
<div class="lm-topbar">
  <div class="lm-brand-wrap">
    <svg width="26" height="26" viewBox="0 0 26 26" fill="none" aria-hidden="true">
      <rect width="26" height="26" rx="7" fill="#0068FF"/>
      <rect x="6.5" y="9" width="2.4" height="8" rx="1.2" fill="#FFFFFF"/>
      <rect x="11.8" y="5.5" width="2.4" height="15" rx="1.2" fill="#FFFFFF"/>
      <rect x="17.1" y="10" width="2.4" height="6" rx="1.2" fill="#FFFFFF"/>
    </svg>
    <div>
      <div class="lm-brand-name">LifeMate <span class="lm-ver-tag">v9.0</span></div>
      <div class="lm-brand-sub">Trợ lý điều phối tác vụ tiếng Việt ngoại tuyến</div>
    </div>
  </div>
  <div class="lm-top-links">
    <a href="{CASE_STUDY_URL}" target="_blank" rel="noopener">Case Study ↗</a>
    <a href="{GITHUB_REPO_URL}" target="_blank" rel="noopener">GitHub ↗</a>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 3 Clean Product Views
# ---------------------------------------------------------------------------
tab_exp, tab_audit, tab_tech = st.tabs(
    ["Trải nghiệm", "Kiểm thử an toàn", "Kỹ thuật & Đánh giá"]
)

# ===========================================================================
# TAB 1: Trải nghiệm (Unified Split-Pane Workbench: Assistant + Live Inspector)
# ===========================================================================
with tab_exp:
    # Read synchronized session state BEFORE running backend turn so both panes match identically
    turn_res = run_workbench_turn(
        st.session_state["workbench_input"],
        scenario_hint=st.session_state.get("workbench_hint"),
        guardrail_enabled=bool(st.session_state.get("workbench_guard", True)),
        wallet_balance=int(st.session_state.get("wb_wallet", 120_000)),
        tomorrow_rain=bool(st.session_state.get("wb_rain", False)),
    )

    col_chat, col_insp = st.columns([1.15, 0.95], gap="large")

    # -----------------------------------------------------------------------
    # Left Pane: End-User Conversation + Unified Composer
    # -----------------------------------------------------------------------
    with col_chat:
        chip_cols = st.columns(len(PRESET_CHIPS))
        for col, (sid, chip_label) in zip(chip_cols, PRESET_CHIPS):
            col.button(
                chip_label,
                key=f"chip_{sid}",
                use_container_width=True,
                on_click=_apply_preset_chip,
                args=(sid,),
            )

        rc = turn_res["receipt_card"]
        rc_cls = {"ok": "lm-receipt-ok", "warn": "lm-receipt-warn", "err": "lm-receipt-err"}.get(rc["kind"], "lm-receipt-ok")
        rc_badge_cls = {"ok": "rc-ok", "warn": "rc-warn", "err": "rc-err"}.get(rc["kind"], "rc-ok")
        mode_caption = "Chế độ bảo vệ V9" if turn_res["guardrail_enabled"] else "Mô hình thô (tắt bảo vệ)"

        chat_html = (
            '<div class="lm-chat-box">'
            '<div class="lm-msg-user-wrap">'
            f'<div class="lm-msg-user">{html.escape(turn_res["utterance"])}</div>'
            "</div>"
            '<div class="lm-msg-bot-head">'
            '<span class="lm-bot-dot">L</span>'
            '<span class="lm-bot-name">LifeMate</span>'
            f'<span class="lm-bot-mode">· {html.escape(mode_caption)}</span>'
            "</div>"
            f'<div class="lm-msg-bot-text">{html.escape(turn_res["assistant_reply"])}</div>'
            f'<div class="lm-receipt {rc_cls}">'
            f'<div class="lm-rc-badge {rc_badge_cls}">{html.escape(rc["badge"])}</div>'
            f'<div class="lm-rc-title">{html.escape(rc["title"])}</div>'
            f'<div class="lm-rc-sub">{html.escape(rc["subtitle"])}</div>'
            "</div>"
            "</div>"
        )
        st.markdown(chat_html, unsafe_allow_html=True)

        with st.form("composer_form", clear_on_submit=False):
            c_inp, c_btn = st.columns([4.4, 1.15], gap="small")
            with c_inp:
                st.text_input(
                    "Nhập câu lệnh tiếng Việt",
                    key="workbench_input",
                    label_visibility="collapsed",
                    placeholder="Gõ câu lệnh tiếng Việt bất kỳ (VD: Chuyển hai triệu rưỡi lúc 8 giờ rưỡi tối mai)...",
                )
            with c_btn:
                st.form_submit_button("Gửi lệnh", use_container_width=True)

    # -----------------------------------------------------------------------
    # Right Pane: Live Inspector (Guard Toggle + Device State + Vertical Rail)
    # -----------------------------------------------------------------------
    with col_insp:
        h_left, h_right = st.columns([1.45, 1.0])
        with h_left:
            st.markdown('<div class="lm-sec-head" style="margin-top:2px;">Trình theo dõi thực thi</div>', unsafe_allow_html=True)
        with h_right:
            st.toggle("Bảo vệ V9", key="workbench_guard")

        active_sid = turn_res.get("scenario_id")
        if active_sid == "fintech_bill_fallback":
            cur_w = int(st.session_state.get("wb_wallet", 120_000))
            st.markdown(
                f'<div style="font-size:0.80rem;color:#475569;margin:4px 0 -4px 0;">Số dư Ví điện tử giả lập: <strong style="color:#0F172A;">{_fmt_vnd(cur_w)}</strong></div>',
                unsafe_allow_html=True,
            )
            st.slider(
                "Số dư Ví điện tử giả lập",
                min_value=0,
                max_value=600_000,
                step=30_000,
                key="wb_wallet",
                label_visibility="collapsed",
            )
        elif active_sid == "conditional_weather_guard":
            st.toggle("Giả lập ngày mai có mưa", key="wb_rain")

        # Live Device State Card
        state_items_html = "".join(
            f'<div class="lm-kv-row">'
            f'<span class="lm-kv-k">{html.escape(r["label"])}</span>'
            f'<div class="lm-kv-r">'
            f'<span class="lm-kv-v">{html.escape(r["value"])}</span>'
            f'<span class="lm-delta d-{r["delta_kind"]}">{html.escape(r["delta"])}</span>'
            f"</div>"
            f"</div>"
            for r in turn_res["state_rows"]
        )
        st.markdown(
            f'<div class="lm-insp-panel"><div class="lm-insp-subhead">Trạng thái dữ liệu</div>{state_items_html}</div>',
            unsafe_allow_html=True,
        )

        # Minimalist Vertical Hairline Execution Rail
        rail_items: list[str] = []
        for s in turn_res["steps"]:
            dot_cls = {"ok": "dot-ok", "warn": "dot-warn", "err": "dot-err"}.get(s["status_kind"], "dot-ok")
            st_color = {"ok": "#059669", "warn": "#D97706", "err": "#DC2626"}.get(s["status_kind"], "#059669")
            args_html = _render_args_chips(s["args"])
            rail_items.append(
                f'<div class="lm-rail-item">'
                f'<span class="lm-rail-dot {dot_cls}"></span>'
                f'<div class="lm-rail-head">'
                f'<span class="lm-rail-tool">{s["idx"]:02d}. {html.escape(s["tool"])}</span>'
                f'<span class="lm-rail-status" style="color:{st_color};">{html.escape(s["status_vi"])}</span>'
                f"</div>"
                f"{args_html}"
                f'<div class="lm-rail-note">{html.escape(s["note_vi"])}</div>'
                f"</div>"
            )
        st.markdown(
            f'<div class="lm-insp-subhead" style="margin-left:2px;">Chuỗi gọi công cụ ({len(turn_res["steps"])} bước)</div>'
            f'<div class="lm-rail">{"".join(rail_items)}</div>',
            unsafe_allow_html=True,
        )

        with st.expander(f"Chi tiết kỹ thuật · Hash {turn_res['initial_hash']} → {turn_res['final_hash']}"):
            st.caption(
                f"Định tuyến Top-K: {', '.join(turn_res['nlp']['recommended_apps'])} "
                f"(giảm {turn_res['nlp']['char_savings_pct']}% ngữ cảnh)"
            )
            st.json(turn_res["final_state"])


# ===========================================================================
# TAB 2: Kiểm thử an toàn (Clean Vietnamese Audit Console)
# ===========================================================================
with tab_audit:
    audit = run_audit_suite_vi()

    a_left, a_right = st.columns([4.3, 1.0])
    with a_left:
        st.markdown(
            f'<div class="lm-summary-bar"><strong>{audit["passed"]} / {audit["total"]} kịch bản đạt an toàn</strong> '
            f'&nbsp;·&nbsp; 0% chặn oan trên hội thoại cũ &nbsp;·&nbsp; Chạy trực tiếp trên CPU ({audit["elapsed_ms"]:.2f} ms)</div>',
            unsafe_allow_html=True,
        )
    with a_right:
        st.button("Chạy lại kiểm thử", key="rerun_audit", use_container_width=True)

    table_rows_html = []
    for row in audit["rows"]:
        badge_txt = "Đạt" if row["safe"] else "Chưa đạt"
        table_rows_html.append(
            "<tr>"
            f'<td><strong>{html.escape(row["title"])}</strong><span class="lm-code-sub">{html.escape(row["code_ref"])}</span></td>'
            f'<td style="color:#475569;">{html.escape(row["risk"])}</td>'
            f'<td>{html.escape(row["v9_handling"])}</td>'
            f'<td><span class="lm-status-pill">● {badge_txt}</span></td>'
            "</tr>"
        )

    st.markdown(
        f"""
<div class="lm-table-wrap">
  <table class="lm-table">
    <thead>
      <tr>
        <th style="width:22%;">Kịch bản kiểm thử</th>
        <th style="width:31%;">Rủi ro khi dùng mô hình thô</th>
        <th style="width:36%;">Cơ chế xử lý của LifeMate V9</th>
        <th style="width:11%;">Kết quả</th>
      </tr>
    </thead>
    <tbody>{"".join(table_rows_html)}</tbody>
  </table>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.expander("Chi tiết cơ chế kiểm soát Schema-First V9"):
        st.markdown(
            "- **Tra cứu trước khi ghi & Tách khóa ngoại:** Chặn mã tự đoán từ giọng nói (V21, SP204) và ngăn nhầm lẫn giữa khóa ngoại (`hotel_id`) với mã đặt chỗ chính (`booking_id`), nhưng vẫn cho phép tái sử dụng đúng ID hợp lệ từ lịch sử hội thoại cũ.\n"
            "- **Ràng buộc suy diễn trực tiếp từ JSON Schema:** Kiểm tra `x-one-of`, giới hạn sức chứa (`max_guests`), khoảng ngày (`check_in < check_out`), thực đơn bắt buộc trước khi đặt món, và điều kiện hoàn đơn.\n"
            "- **Bảo vệ giao dịch tài chính xác định:** Đối chiếu số dư thật giữa Ví điện tử và Ngân hàng trước khi ghi nhận thanh toán; tự động chặn SMS báo thành công giả nếu giao dịch thất bại."
        )


# ===========================================================================
# TAB 3: Kỹ thuật & Đánh giá (Editorial Report + Plotly No-Toolbar + HTML Tables)
# ===========================================================================
with tab_tech:
    st.markdown(
        f"""
<div class="lm-ed-bar">
  <div class="lm-ed-item">
    <div class="lm-ed-k">Public Leaderboard (V7)</div>
    <div class="lm-ed-v">49.60</div>
    <div class="lm-ed-sub">+23.80 so với Baseline 25.80</div>
  </div>
  <div class="lm-ed-item">
    <div class="lm-ed-k">Private Evaluation</div>
    <div class="lm-ed-v">{PRIVATE_BENCHMARK_METRICS['final_private']:.2f}</div>
    <div class="lm-ed-sub">+{PRIVATE_BENCHMARK_METRICS['asr_debug_delta']:.2f} sau sửa lỗi nạp ASR</div>
  </div>
  <div class="lm-ed-item">
    <div class="lm-ed-k">Ngân sách tham số</div>
    <div class="lm-ed-v">4.8B</div>
    <div class="lm-ed-sub">Qwen3-4B + Whisper (&lt; 6B Offline)</div>
  </div>
  <div class="lm-ed-item">
    <div class="lm-ed-k">Huấn luyện L4 (9.216 tok)</div>
    <div class="lm-ed-v">{L4_PEAK_CHUNKED_VRAM_GIB} GiB</div>
    <div class="lm-ed-sub">Giới hạn {L4_TOTAL_VRAM_GIB} GiB · Loss 0.01953</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    c_left, c_right = st.columns(2, gap="medium")
    with c_left:
        df_prog = pd.DataFrame(SCORE_PROGRESSION)
        fig_s = px.line(
            df_prog,
            x="version",
            y="score",
            markers=True,
            text="score",
            title="Tiến trình điểm Public qua các phiên bản",
            color_discrete_sequence=["#0068FF"],
        )
        fig_s.update_traces(line=dict(width=2.6), marker=dict(size=7, color="#0068FF"), textposition="top center")
        fig_s.update_yaxes(range=[20, 56], title="Điểm Public")
        fig_s.update_xaxes(title="")
        st.plotly_chart(_plotly_editorial(fig_s, height=285), use_container_width=True, config={"displayModeBar": False})

    with c_right:
        df_loss = pd.DataFrame(V7_LOSS_CURVE)
        fig_l = go.Figure()
        fig_l.add_trace(
            go.Scatter(
                x=df_loss["step"],
                y=df_loss["train_loss"],
                mode="lines+markers",
                name="Train Loss",
                line=dict(color="#94A3B8", width=2),
                marker=dict(size=5),
            )
        )
        fig_l.add_trace(
            go.Scatter(
                x=df_loss["step"],
                y=df_loss["val_loss"],
                mode="lines+markers",
                name="Validation Loss (201 mẫu)",
                line=dict(color="#0068FF", width=2.8),
                marker=dict(size=6),
            )
        )
        fig_l.add_hline(
            y=V6_BEST_VAL_LOSS,
            line_dash="dot",
            line_color="#64748B",
            annotation_text=f"Mốc V6 ({V6_BEST_VAL_LOSS})",
            annotation_position="bottom left",
        )
        fig_l.update_layout(
            title="Hội tụ SFT V7 trên 1 GPU L4 (9.216 token)",
            xaxis_title="Step",
            yaxis_title="Cross-Entropy Loss",
            legend=dict(
                x=0.98,
                y=0.96,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="#E6E8EC",
                borderwidth=1,
            ),
        )
        fig_l = _plotly_editorial(fig_l, height=285)
        fig_l.update_layout(margin=dict(l=34, r=14, t=48, b=32))
        st.plotly_chart(fig_l, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="lm-sec-head">So sánh tiến hóa hậu cuộc thi: V7 → V9</div>', unsafe_allow_html=True)
    evo_rows_html = []
    for er in _cached_v9_evolution():
        evo_rows_html.append(
            "<tr>"
            f'<td>{html.escape(er["Hạng mục kiểm thử (Synthetic CPU)"])}</td>'
            f'<td class="num" style="color:#64748B;">{html.escape(er["LLM Thô"])}</td>'
            f'<td class="num" style="color:#475569;">{html.escape(er["LifeMate V7"])}</td>'
            f'<td class="num v9col">{html.escape(er["LifeMate V9"])}</td>'
            "</tr>"
        )
    st.markdown(
        f"""
<div class="lm-table-wrap">
  <table class="lm-table">
    <thead>
      <tr>
        <th style="width:46%;">Hạng mục kiểm thử (Bộ tổng hợp CPU)</th>
        <th class="num" style="width:18%;">LLM Thô</th>
        <th class="num" style="width:18%;">LifeMate V7</th>
        <th class="num v9col" style="width:18%;">LifeMate V9</th>
      </tr>
    </thead>
    <tbody>{"".join(evo_rows_html)}</tbody>
  </table>
</div>
<div class="lm-caption">Tái lập trực tiếp trên CPU bằng lệnh: <code>python3 scripts/benchmarks/benchmark_v9_suite.py</code></div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="lm-sec-head">Kiến trúc 5 tầng &amp; Tối ưu bộ nhớ GPU</div>', unsafe_allow_html=True)
    arch_rows_html = []
    for layer in get_architecture_layers_vi():
        arch_rows_html.append(
            "<tr>"
            f'<td style="font-weight:600;white-space:nowrap;">{html.escape(layer["layer"])}</td>'
            f'<td class="mono" style="font-size:0.78rem;color:#334155;">{html.escape(layer["component"])}</td>'
            f'<td style="color:#475569;">{html.escape(layer["role"])}</td>'
            "</tr>"
        )
    st.markdown(
        f"""
<div class="lm-table-wrap">
  <table class="lm-table">
    <thead>
      <tr>
        <th style="width:22%;">Tầng xử lý</th>
        <th style="width:32%;">Thành phần cốt lõi</th>
        <th style="width:46%;">Tác dụng thực tế</th>
      </tr>
    </thead>
    <tbody>{"".join(arch_rows_html)}</tbody>
  </table>
</div>
<div class="lm-caption">Chi tiết thiết kế hệ thống &amp; phân tích lỗi hậu cuộc thi: <a href="{CASE_STUDY_URL}" target="_blank" rel="noopener" style="color:#0068FF;text-decoration:none;font-weight:500;">docs/ARCHITECTURE_CASE_STUDY.md ↗</a></div>
""",
        unsafe_allow_html=True,
    )
