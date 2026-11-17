"""LifeMate — Minimalist Vietnamese Offline Agent & V8 Safety Workspace.

Designed around clean product clarity (zero visual noise, fast response, immediate utility)
while executing the genuine Python core pipeline (`src/`) on CPU.
"""
from __future__ import annotations

import html
import sys
from datetime import datetime
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
    FULL_CATALOG_APPS,
    L4_PEAK_CHUNKED_VRAM_GIB,
    L4_TOTAL_VRAM_GIB,
    SCENARIO_CATALOG,
    SCORE_PROGRESSION,
    V6_BEST_VAL_LOSS,
    V7_LOSS_CURVE,
    analyze_vietnamese_utterance,
    run_fintech_chaos_matrix,
    run_sandbox_scenario,
)

st.set_page_config(
    page_title="LifeMate — Trợ lý tác vụ tiếng Việt ngoại tuyến",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Clean Zalo-Inspired Daylight Design System (#FFFFFF / #F7F8FA / #0068FF)
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-canvas: #FFFFFF;
    --bg-surface: #F7F8FA;
    --border-hairline: #E5E7EB;
    --border-strong: #CBD5E1;
    --text-primary: #111827;
    --text-secondary: #4B5563;
    --text-muted: #6B7280;
    --brand-blue: #0068FF;
    --status-ok: #059669;
    --status-ok-bg: #ECFDF5;
    --status-warn: #B45309;
    --status-warn-bg: #FFFBEB;
    --status-err: #DC2626;
    --status-err-bg: #FEF2F2;
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background-color: var(--bg-canvas);
    color: var(--text-primary);
}

code, pre, .mono {
    font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
}

/* Remove sidebar chrome completely — keep user focused on 1 clean workspace */
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
#MainMenu,
footer {
    display: none !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
    height: 2.4rem !important;
}

.block-container,
[data-testid="stMainBlockContainer"] {
    max-width: 1060px;
    margin: 0 auto;
    padding: 3.4rem 1.5rem 2.75rem 1.5rem !important;
}

/* Top Product Header: clean, restrained, zero gradient or badge soup */
.lm-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    flex-wrap: wrap;
    gap: 12px;
    padding-bottom: 14px;
    margin-bottom: 14px;
    border-bottom: 1px solid var(--border-hairline);
}
.lm-brand-title {
    font-size: 1.45rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: var(--text-primary);
    line-height: 1.2;
}
.lm-brand-sub {
    font-size: 0.875rem;
    color: var(--text-secondary);
    margin-top: 3px;
}
.lm-meta-strip {
    font-size: 0.80rem;
    color: var(--text-muted);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}
.lm-meta-strip strong {
    color: var(--text-primary);
    font-weight: 600;
}

/* Underlined editorial tabs */
[data-baseweb="tab-list"] {
    gap: 24px;
    border-bottom: 1px solid var(--border-hairline);
    margin-bottom: 18px;
}
[data-baseweb="tab"] {
    padding: 8px 2px 10px 2px !important;
    font-size: 0.90rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    background: transparent !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: var(--brand-blue) !important;
    font-weight: 600 !important;
}

/* Flat metric cells */
.lm-strip {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    border: 1px solid var(--border-hairline);
    border-radius: 8px;
    background: var(--bg-surface);
    margin: 12px 0 18px 0;
    overflow: hidden;
}
.lm-cell {
    padding: 12px 15px;
    border-right: 1px solid var(--border-hairline);
}
.lm-cell:last-child {
    border-right: none;
}
.lm-cell-label {
    font-size: 0.74rem;
    font-weight: 500;
    color: var(--text-muted);
    margin-bottom: 4px;
}
.lm-cell-val {
    font-size: 1.12rem;
    font-weight: 600;
    color: var(--text-primary);
    font-variant-numeric: tabular-nums;
    line-height: 1.3;
    word-break: break-word;
}
.lm-cell-sub {
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-top: 3px;
}

/* Quick preset buttons clean polish */
div[data-testid="stButton"] button {
    border-radius: 6px;
    border: 1px solid var(--border-hairline);
    background-color: var(--bg-surface);
    color: var(--text-primary);
    font-size: 0.80rem;
    font-weight: 500;
    padding: 0.42rem 0.65rem;
    transition: border-color 0.12s ease, background-color 0.12s ease;
}
div[data-testid="stButton"] button:hover {
    border-color: var(--brand-blue);
    background-color: #F0F6FF;
    color: var(--brand-blue);
}

/* Execution step rows */
.lm-step {
    border: 1px solid var(--border-hairline);
    border-radius: 8px;
    padding: 10px 13px;
    margin-bottom: 8px;
    background: var(--bg-canvas);
}
.lm-step-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 5px;
}
.lm-step-idx {
    font-size: 0.76rem;
    font-weight: 600;
    color: var(--text-muted);
    font-variant-numeric: tabular-nums;
}
.lm-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.70rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.tag-ok { color: var(--status-ok); background: var(--status-ok-bg); }
.tag-warn { color: var(--status-warn); background: var(--status-warn-bg); }
.tag-err { color: var(--status-err); background: var(--status-err-bg); }
.lm-call {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 0.78rem;
    color: var(--text-primary);
    background: var(--bg-surface);
    padding: 5px 8px;
    border-radius: 5px;
    word-break: break-all;
}
.lm-note {
    font-size: 0.80rem;
    color: var(--text-secondary);
    margin-top: 5px;
    line-height: 1.4;
}

/* State summary card */
.lm-state-card {
    border: 1px solid var(--border-hairline);
    border-radius: 8px;
    background: var(--bg-surface);
    padding: 14px 16px;
}
.lm-state-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 12px;
    padding: 7px 0;
    border-bottom: 1px solid var(--border-hairline);
    font-size: 0.84rem;
}
.lm-state-row:last-child {
    border-bottom: none;
}
.lm-state-k {
    color: var(--text-secondary);
}
.lm-state-v {
    color: var(--text-primary);
    font-weight: 600;
    text-align: right;
    font-variant-numeric: tabular-nums;
}

/* Flat 3-col defense cards */
.lm-pillar-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-top: 14px;
}
.lm-pillar {
    border: 1px solid var(--border-hairline);
    border-radius: 8px;
    background: var(--bg-surface);
    padding: 13px 15px;
}
.lm-pillar h4 {
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 5px 0;
}
.lm-pillar p {
    font-size: 0.80rem;
    color: var(--text-secondary);
    margin: 0;
    line-height: 1.45;
}

.lm-section-title {
    font-size: 1.02rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 2px 0 4px 0;
}
.lm-section-desc {
    font-size: 0.84rem;
    color: var(--text-secondary);
    margin-bottom: 10px;
}

@media (max-width: 820px) {
    .lm-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .lm-pillar-grid { grid-template-columns: 1fr; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Formatting & Domain Helpers
# ---------------------------------------------------------------------------
SCENARIO_UI_META: dict[str, dict[str, str]] = {
    "fintech_bill_fallback": {
        "btn": "Trả tiền điện (thiếu ví)",
        "label": "Hóa đơn EVN — Đổi nguồn khi ví thiếu tiền",
    },
    "v8_food_menu_dependency": {
        "btn": "Đặt phở theo thực đơn",
        "label": "Giao đồ ăn — Bắt buộc đọc thực đơn trước khi đặt (V8)",
    },
    "spoken_id_read_before_write": {
        "btn": "Thêm giỏ hàng (lọc mã giả)",
        "label": "Đi chợ — Chặn mã đọc miệng 'SP204' chưa tra cứu",
    },
    "conditional_weather_guard": {
        "btn": "Nhắc lịch theo thời tiết",
        "label": "Thời tiết — Chỉ tạo nhắc nhở nếu ngày mai mưa",
    },
}

PRESET_EXTRA_INPUTS = {
    "chuyen_tien": "Chuyển hai triệu năm trăm nghìn cho chị Lan lúc 8 giờ rưỡi tối mai nhé",
}


def _fmt_vnd(amount: int | float | None) -> str:
    if amount is None:
        return "Không có"
    return f"{int(amount):,}".replace(",", ".") + " đ"


def _fmt_iso_vi(iso_str: str | None) -> tuple[str, str]:
    if not iso_str:
        return "Không xác định", "Chưa có mốc giờ trong câu"
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%d/%m/%Y · %H:%M"), f"ISO +07:00 ({dt.strftime('%Y-%m-%dT%H:%M')})"
    except ValueError:
        return iso_str, "Định dạng ISO-8601"


def _render_metric_strip(cells: list[tuple[str, str, str]]) -> None:
    parts = ['<div class="lm-strip">']
    for label, val, sub in cells:
        parts.append(
            f'<div class="lm-cell">'
            f'<div class="lm-cell-label">{html.escape(label)}</div>'
            f'<div class="lm-cell-val">{html.escape(val)}</div>'
            f'<div class="lm-cell-sub">{html.escape(sub)}</div>'
            f"</div>"
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _step_badge_and_note(step: dict[str, Any]) -> tuple[str, str, str]:
    status = step["status"]
    raw_note = str(step.get("guardrail_note", ""))
    mapping = {
        "EXECUTED": ("Hợp lệ", "tag-ok", "Đã qua kiểm tra schema và trạng thái hệ thống."),
        "EXECUTED_UNCHECKED": ("Chạy thẳng", "tag-warn", "Thực thi trực tiếp vì đã tắt lá chắn bảo vệ."),
        "INTERCEPTED": ("Đã chặn", "tag-warn", raw_note),
        "AUTO_CORRECTED": ("Đổi nguồn tiền", "tag-warn", "Ví không đủ số dư, tự động chuyển sang tài khoản ngân hàng."),
        "STOPPED_BY_CONDITION": ("Dừng an toàn", "tag-ok", "Điều kiện thời tiết không thỏa (không mưa), không tạo nhắc nhở thừa."),
        "SOFT_NOOP": ("Lỗi Soft-Noop", "tag-err", raw_note),
        "MISSED": ("Bỏ sót lệnh", "tag-err", "Logic cũ hiểu nhầm câu điều kiện và hủy toàn bộ yêu cầu từ đầu."),
    }
    label, cls, note = mapping.get(status, (status, "tag-warn", raw_note))
    if status == "INTERCEPTED" and "food_delivery.get_menu" in raw_note:
        note = "Chặn đặt món vì chưa đọc thực đơn (food_delivery.get_menu) để lấy item_id thật."
    elif status == "INTERCEPTED" and "grocery.search" in raw_note:
        note = "Chặn mã đọc miệng 'SP204' vì chưa tra mã sản phẩm thật qua grocery.search."
    return label, cls, note


def _summarize_state_rows(res: dict[str, Any]) -> list[tuple[str, str]]:
    final_state = res["final_state"]
    rows: list[tuple[str, str]] = []

    if "bills" in final_state and final_state["bills"]:
        bill = final_state["bills"][0]
        status_map = {
            "unpaid": "Chưa thanh toán",
            "paid_via_ewallet": "Đã trả qua Ví điện tử",
            "paid_via_bank": "Đã trả qua Ngân hàng",
        }
        rows.append(("Hóa đơn điện EVN", status_map.get(bill.get("status", ""), bill.get("status", ""))))
    if "ewallet_balance" in final_state:
        rows.append(("Số dư Ví điện tử", _fmt_vnd(final_state["ewallet_balance"])))
    if "bank_balance" in final_state:
        rows.append(("Số dư Ngân hàng", _fmt_vnd(final_state["bank_balance"])))
    if "sent_sms" in final_state:
        sms_list = final_state["sent_sms"]
        rows.append(("Tin nhắn SMS đã gửi", f"{len(sms_list)} tin" if sms_list else "0 tin"))
    if "food_orders" in final_state:
        orders = final_state["food_orders"]
        rows.append(("Đơn đặt đồ ăn", f"{len(orders)} đơn hợp lệ" if orders else "Chưa tạo được đơn"))
    if "cart" in final_state:
        cart = final_state["cart"]
        rows.append(("Giỏ hàng siêu thị", f"{len(cart)} mặt hàng hợp lệ" if cart else "Trống (chưa thêm được)"))
    if "reminders" in final_state:
        rems = final_state["reminders"]
        rows.append(("Nhắc nhở đã tạo", f"{len(rems)} nhắc nhở" if rems else "Không tạo nhắc nhở"))

    rows.append(("Thay đổi trạng thái (Hash)", f"{res['initial_hash']} -> {res['final_hash']}"))
    return rows


def _plotly_clean_light(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#111827", family="Inter, sans-serif", size=12),
        title_font=dict(size=14, color="#111827"),
        height=height,
        margin=dict(l=32, r=16, t=42, b=32),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E5E7EB", zeroline=False)
    fig.update_yaxes(gridcolor="#F1F5F9", linecolor="#E5E7EB", zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# Minimalist Product Header
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="lm-header">
  <div>
    <div class="lm-brand-title">LifeMate</div>
    <div class="lm-brand-sub">Trợ lý điều phối tác vụ tiếng Việt ngoại tuyến — Thực thi đúng lệnh, bảo vệ giao dịch trước khi chạm API</div>
  </div>
  <div class="lm-meta-strip">
    Public Score <strong>49.60</strong> &nbsp;·&nbsp; Mô hình <strong>4.8B (&lt; 6B)</strong> &nbsp;·&nbsp; Chế độ chạy <strong>CPU Offline</strong>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session defaults for smooth instant interaction
# ---------------------------------------------------------------------------
if "active_scenario" not in st.session_state:
    st.session_state["active_scenario"] = "fintech_bill_fallback"
if "user_utterance" not in st.session_state:
    st.session_state["user_utterance"] = SCENARIO_CATALOG["fintech_bill_fallback"]["utterance"]


def _apply_preset(scenario_id: str) -> None:
    st.session_state["active_scenario"] = scenario_id
    st.session_state["user_utterance"] = SCENARIO_CATALOG[scenario_id]["utterance"]


# ---------------------------------------------------------------------------
# 3 Focused Product Tabs
# ---------------------------------------------------------------------------
tab_try, tab_safety, tab_arch = st.tabs(
    ["Thử nghiệm Trợ lý", "An toàn Giao dịch", "Hiệu năng & Kiến trúc"]
)

# ===========================================================================
# TAB 1: Thử nghiệm Trợ lý (Hữu ích ngay trong 3 giây cho mọi người dùng)
# ===========================================================================
with tab_try:
    st.markdown('<div class="lm-section-title">1. Thử câu lệnh tiếng Việt tự nhiên</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="lm-section-desc">Chọn nhanh 1 tình huống bên dưới hoặc nhập câu nói bất kỳ (có thể viết số tiền bằng chữ, giờ tương đối như "8 giờ rưỡi tối mai").</div>',
        unsafe_allow_html=True,
    )

    p_cols = st.columns(4)
    for col, sid in zip(p_cols, list(SCENARIO_UI_META.keys())):
        if col.button(SCENARIO_UI_META[sid]["btn"], key=f"btn_{sid}", use_container_width=True):
            _apply_preset(sid)

    utterance_input = st.text_input(
        "Câu lệnh tiếng Việt",
        key="user_utterance",
        label_visibility="collapsed",
        placeholder="Ví dụ: Chuyển hai triệu năm trăm nghìn cho chị Lan lúc 8 giờ rưỡi tối mai nhé",
    )

    nlp_out = analyze_vietnamese_utterance(utterance_input or "Mở ví điện tử", top_k=4)
    top_app_id = nlp_out["recommended_apps"][0] if nlp_out["recommended_apps"] else None
    top_app_vi = FULL_CATALOG_APPS[top_app_id]["name_vi"] if top_app_id in FULL_CATALOG_APPS else "Chưa khớp"
    next_app_id = nlp_out["recommended_apps"][1] if len(nlp_out["recommended_apps"]) > 1 else None
    next_app_sub = f"Ưu tiên kế: {FULL_CATALOG_APPS[next_app_id]['name_vi']}" if next_app_id in FULL_CATALOG_APPS else "Khớp trực tiếp TF-IDF"

    date_main, date_sub = _fmt_iso_vi(nlp_out["resolved_iso"])
    monetary_val = nlp_out.get("monetary_amount")
    amt_str = _fmt_vnd(monetary_val)
    amt_sub = "Trích xuất số tiền tiếng Việt" if monetary_val is not None else "Không chứa số tiền"

    _render_metric_strip(
        [
            ("Ứng dụng định tuyến", top_app_vi, next_app_sub),
            ("Số tiền nhận diện", amt_str, amt_sub),
            ("Thời gian hẹn", date_main, date_sub),
            ("Nén ngữ cảnh đầu vào", f"-{nlp_out['char_savings_pct']}%", f"Chọn Top-4 / {len(FULL_CATALOG_APPS)} ứng dụng"),
        ]
    )

    with st.expander("Xem chi tiết điểm xếp hạng ứng dụng (TF-IDF Semantic Router)"):
        df_rank = pd.DataFrame(nlp_out["ranked_rows"]).sort_values("score", ascending=True)
        fig_r = px.bar(
            df_rank,
            x="score",
            y="name_vi",
            color="tier",
            orientation="h",
            color_discrete_map={"Recommended (Top-K)": "#0068FF", "Secondary": "#CBD5E1"},
        )
        fig_r.update_layout(showlegend=True, legend_title_text="", title="Độ tương đồng giữa câu lệnh và từng ứng dụng")
        fig_r.update_xaxes(title="Điểm tương đồng TF-IDF")
        fig_r.update_yaxes(title="")
        st.plotly_chart(_plotly_clean_light(fig_r, height=290), use_container_width=True)

    st.markdown('<div style="border-top:1px solid #E5E7EB;margin:10px 0 14px 0;"></div>', unsafe_allow_html=True)

    # Section 2: Interactive Sandbox Execution (contextual controls only!)
    st.markdown('<div class="lm-section-title">2. Mô phỏng thực thi & Lá chắn bảo vệ (Sandbox)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="lm-section-desc">Đối chiếu quá trình gọi API của mô hình khi <b>bật</b> hoặc <b>tắt</b> lá chắn V8 (hãy thử tắt lá chắn để thấy lỗi mất tiền / sai đơn hàng thực tế).</div>',
        unsafe_allow_html=True,
    )

    c_scen, c_ctx, c_guard = st.columns([2.2, 1.3, 1.0])
    with c_scen:
        selected_scen = st.selectbox(
            "Tình huống mô phỏng",
            list(SCENARIO_CATALOG.keys()),
            key="active_scenario",
            format_func=lambda k: SCENARIO_UI_META[k]["label"],
        )

    wallet_val = 120_000
    rain_val = False
    with c_ctx:
        if selected_scen == "fintech_bill_fallback":
            wallet_val = st.slider(
                "Số dư Ví điện tử (đ)",
                min_value=0,
                max_value=600_000,
                value=120_000,
                step=30_000,
            )
        elif selected_scen == "conditional_weather_guard":
            rain_val = st.toggle("Giả lập ngày mai trời mưa", value=False)
        else:
            st.markdown(
                "<div style='padding-top:28px;font-size:0.82rem;color:#4B5563;'>Kiểm tra ràng buộc Schema &amp; ID trước khi gọi API</div>",
                unsafe_allow_html=True,
            )

    with c_guard:
        st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
        guardrail_on = st.toggle("Bật lá chắn V8", value=True)

    sb_res = run_sandbox_scenario(
        selected_scen,
        guardrail_enabled=guardrail_on,
        wallet_balance=wallet_val,
        tomorrow_rain=rain_val,
    )

    left_col, right_col = st.columns([1.3, 1.0])
    with left_col:
        st.caption(f"Nhật ký gọi công cụ — Số lệnh thực thi: {len(sb_res['executed_calls'])} · Lỗi Soft-Noop: {sb_res['soft_noops']}")
        for step in sb_res["trace_steps"]:
            tag_label, tag_cls, note_vi = _step_badge_and_note(step)
            st.markdown(
                f"""
                <div class="lm-step">
                  <div class="lm-step-top">
                    <span class="lm-step-idx">Bước {step['step']:02d}</span>
                    <span class="lm-tag {tag_cls}">{html.escape(tag_label)}</span>
                  </div>
                  <div class="lm-call">{html.escape(step['proposed_call'])}</div>
                  <div class="lm-note">{html.escape(note_vi)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right_col:
        st.caption("Trạng thái dữ liệu sau khi chạy (Trước -> Sau)")
        summary_rows = _summarize_state_rows(sb_res)
        row_html = [
            f'<div class="lm-state-row"><span class="lm-state-k">{html.escape(k)}</span><span class="lm-state-v">{html.escape(v)}</span></div>'
            for k, v in summary_rows
        ]
        st.markdown(f'<div class="lm-state-card">{"".join(row_html)}</div>', unsafe_allow_html=True)

        with st.expander("Xem JSON trạng thái thô (dành cho kỹ sư)"):
            st.json(sb_res["final_state"])

# ===========================================================================
# TAB 2: An toàn Giao dịch
# ===========================================================================
with tab_safety:
    st.markdown('<div class="lm-section-title">Kiểm chứng 5 tình huống rủi ro khi chạy mô hình nhỏ (&lt; 6B)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="lm-section-desc">Toàn bộ kịch bản bên dưới chạy trực tiếp qua <code>PureSchemaStateTracker</code> và <code>HybridProgramAidedEngine</code> trên CPU ngay khi tải trang.</div>',
        unsafe_allow_html=True,
    )

    matrix_raw = run_fintech_chaos_matrix()
    passed = sum(1 for row in matrix_raw if row["Safe"])

    _render_metric_strip(
        [
            ("Tỉ lệ chặn lỗi thực nghiệm", f"{passed}/{len(matrix_raw)} đạt", "Chạy sống từ src/guardrail.py"),
            ("Độ chính xác chuyển nguồn ví", "100%", "Ví thiếu tiền tự động qua Ngân hàng"),
            ("Mã nói miệng chưa tra cứu", "0 lọt lưới", "Chặn SP204 / V21 trước khi ghi"),
            ("Kiểm tra thực đơn trước đặt món", "Bắt buộc (V8)", "Chặn place_order trước get_menu"),
        ]
    )

    vi_table = pd.DataFrame(
        [
            {
                "Tình huống": {
                    "Weather Conditional": "Câu điều kiện thời tiết",
                    "Spoken ID ('V21')": "Mã đọc miệng ('V21')",
                    "Food Menu Gate": "Đặt món chưa xem menu",
                    "Return Status Gate": "Hoàn đơn đã bị hủy",
                    "Wallet Fallback": "Thanh toán khi ví thiếu tiền",
                }.get(r["Scenario"], r["Scenario"]),
                "Lỗ hổng mục tiêu": r["Threat"],
                "LLM không bảo vệ": r["Raw LLM"],
                "LifeMate V8": r["V8 Guardrail"],
                "Trạng thái": "An toàn" if r["Safe"] else "Thất bại",
            }
            for r in matrix_raw
        ]
    )
    st.dataframe(vi_table, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="lm-pillar-grid">
          <div class="lm-pillar">
            <h4>1. Tra cứu trước khi ghi</h4>
            <p>Không cho phép mô hình tự điền mã sản phẩm, mã đơn hay mã bàn từ lời nói người dùng (như V21, SP204) nếu mã đó chưa từng xuất hiện từ lệnh đọc API.</p>
          </div>
          <div class="lm-pillar">
            <h4>2. Thứ tự phụ thuộc nghiệp vụ (V8)</h4>
            <p>Chặn ngay lệnh đặt món <code>food_delivery.place_order</code> nếu chưa đọc <code>get_menu</code>, đồng thời khóa quyền hoàn đơn với đơn đã <code>cancelled</code> hoặc đang giao.</p>
          </div>
          <div class="lm-pillar">
            <h4>3. Lớp bảo vệ tài chính xác định</h4>
            <p>Kiểm tra số dư thật giữa Ví và Ngân hàng trước khi trừ tiền; nếu giao dịch lỗi do thiếu số dư thì chặn luôn tin nhắn SMS báo thành công giả.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===========================================================================
# TAB 3: Hiệu năng & Kiến trúc
# ===========================================================================
with tab_arch:
    st.markdown('<div class="lm-section-title">Tối ưu mô hình 4.8B & Huấn luyện 9.216 token trên 1 GPU L4</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="lm-section-desc">Giải bài toán thực tế với ngân sách tham số &lt; 6B, không kết nối mạng ngoài (<code>--network none</code>) và bộ nhớ GPU giới hạn 22,03 GiB.</div>',
        unsafe_allow_html=True,
    )

    _render_metric_strip(
        [
            ("Public Leaderboard V7", "49.60", "+23.80 điểm so với Baseline 25.80"),
            ("Ngân sách tham số", "4.8B", "Qwen3-4B (4.0B) + Whisper (0.8B)"),
            ("Best Eval Loss (V7)", "0.01953", "Hội tụ tại Step 250 (LoRA r=128)"),
            ("VRAM đỉnh khi train L4", f"{L4_PEAK_CHUNKED_VRAM_GIB} GiB", f"An toàn trong ngưỡng {L4_TOTAL_VRAM_GIB} GiB"),
        ]
    )

    c_left, c_right = st.columns(2)
    with c_left:
        df_prog = pd.DataFrame(SCORE_PROGRESSION)
        fig_s = px.line(
            df_prog,
            x="version",
            y="score",
            markers=True,
            text="score",
            title="Tiến trình điểm số qua các phiên bản",
            color_discrete_sequence=["#0068FF"],
        )
        fig_s.update_traces(line=dict(width=2.8), marker=dict(size=8, color="#0068FF"), textposition="top center")
        fig_s.update_yaxes(range=[20, 56], title="Public Score")
        fig_s.update_xaxes(title="")
        st.plotly_chart(_plotly_clean_light(fig_s, height=300), use_container_width=True)

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
                line=dict(color="#0068FF", width=3),
                marker=dict(size=6),
            )
        )
        fig_l.add_hline(y=V6_BEST_VAL_LOSS, line_dash="dot", line_color="#6B7280", annotation_text=f"Mốc V6 ({V6_BEST_VAL_LOSS})")
        fig_l.update_layout(
            title="Hội tụ SFT V7 (9.216 token)",
            xaxis_title="Step",
            yaxis_title="Cross-Entropy Loss",
            legend=dict(
                x=0.98,
                y=0.96,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor="#E5E7EB",
                borderwidth=1,
            ),
        )
        fig_l = _plotly_clean_light(fig_l, height=300)
        fig_l.update_layout(margin=dict(l=36, r=16, t=54, b=36))
        st.plotly_chart(fig_l, use_container_width=True)

    st.markdown("**Kiến trúc xử lý 5 tầng & Tối ưu bộ nhớ**")
    df_arch = pd.DataFrame(
        [
            {"Tầng xử lý": "1. Nhận dạng giọng nói (Docker GPU)", "Thành phần": "Whisper Turbo CT2 (~0.8B) + Dual-ASR Fallback", "Tác dụng cốt lõi": "Loại nhiễu âm thanh, tránh ảo giác lặp từ đầu vào"},
            {"Tầng xử lý": "2. Chuẩn hóa tiếng Việt", "Thành phần": "SafeContextPreprocessor & Number/Date Engine", "Tác dụng cốt lõi": "Dịch số ghép ('hai triệu rưỡi'), giờ tương đối sang ISO +07:00"},
            {"Tầng xử lý": "3. Định tuyến công cụ", "Thành phần": "SemanticToolRouter (Word 1,2-gram TF-IDF)", "Tác dụng cốt lõi": "Xếp hạng Top-K trong 57 ứng dụng < 2ms, không cần embedding GPU"},
            {"Tầng xử lý": "4. Lập luận & Gọi API", "Thành phần": "Qwen3-4B + LoRA (r=128) Chunked-CE (0.15 GiB CE)", "Tác dụng cốt lõi": "Giảm tiêu thụ CE Logits từ 5.30 GiB xuống 0.15 GiB ở 9.216 token"},
            {"Tầng xử lý": "5. Lá chắn thực thi V8", "Thành phần": "PureSchemaStateTracker + HybridEngine", "Tác dụng cốt lõi": "Chặn ID chưa tra cứu, kiểm tra menu & số dư trước khi chạm API"},
        ]
    )
    st.dataframe(df_arch, use_container_width=True, hide_index=True)
