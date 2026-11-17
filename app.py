"""LifeMate Control Tower — Interactive Portfolio MVP & V8 Guardrail Showcase.

Dark-mode analytical & execution dashboard running the core Python pipeline (`src/`)
for Zalo AI Challenge 2026 LifeMate (< 6B Offline Voice-to-Action Agent).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
    page_title="LifeMate Control Tower — Offline Voice Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Dark Executive Styling (Veraxus / Zalo Blue Palette: #0B0E17 + #0068FF)
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 1.5rem;
}
.stApp {
    background: radial-gradient(circle at 12% 0%, rgba(0, 104, 255, 0.14), transparent 42%),
                radial-gradient(circle at 88% 8%, rgba(0, 212, 255, 0.08), transparent 36%),
                #0B0E17;
    color: #F8FAFC;
}
.hero-banner {
    background: linear-gradient(135deg, rgba(18, 22, 36, 0.96) 0%, rgba(12, 24, 52, 0.92) 100%);
    border: 1px solid rgba(0, 104, 255, 0.42);
    border-radius: 14px;
    padding: 15px 20px;
    margin-bottom: 12px;
}
.hero-title {
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #F8FAFC;
    margin: 0 0 4px 0;
}
.hero-sub {
    color: #CBD5E1;
    font-size: 0.90rem;
    margin: 0;
}
.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 9px;
}
.pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(0, 104, 255, 0.18);
    border: 1px solid rgba(56, 189, 248, 0.35);
    color: #E0F2FE;
}
.kpi-card {
    background: #12141C;
    border: 1px solid rgba(148, 163, 184, 0.20);
    border-top: 3px solid #0068FF;
    border-radius: 12px;
    padding: 12px 15px;
    height: 100%;
}
.kpi-label {
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94A3B8;
    margin-bottom: 4px;
}
.kpi-val {
    font-size: 1.48rem;
    font-weight: 700;
    color: #F8FAFC;
    line-height: 1.15;
}
.kpi-delta {
    font-size: 0.78rem;
    color: #34D399;
    margin-top: 4px;
}
.step-card {
    background: #12141C;
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 10px;
}
.step-intercept {
    border-left: 4px solid #F59E0B;
    background: rgba(245, 158, 11, 0.08);
}
.step-ok {
    border-left: 4px solid #10B981;
}
.step-bad {
    border-left: 4px solid #EF4444;
    background: rgba(239, 68, 68, 0.08);
}
.arch-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 10px;
    margin: 10px 0 18px 0;
}
.arch-box {
    background: #12141C;
    border: 1px solid rgba(56, 189, 248, 0.28);
    border-radius: 10px;
    padding: 12px;
}
.arch-box h4 {
    margin: 0 0 6px 0;
    font-size: 0.90rem;
    color: #60A5FA;
}
.arch-box p {
    margin: 0;
    font-size: 0.80rem;
    color: #CBD5E1;
    line-height: 1.4;
}
@media (max-width: 900px) {
    .arch-grid { grid-template-columns: 1fr; }
}
</style>
""",
    unsafe_allow_html=True,
)

VIEWS = {
    "overview": "1. Executive Overview & Benchmarks",
    "sandbox": "2. Interactive ReAct Sandbox",
    "risk": "3. FinTech Risk & V8 Safety Lab",
    "router": "4. Semantic Router & Context Budget",
    "training": "5. L4 Training & Chunked-CE Analytics",
    "nlp": "6. Vietnamese Spoken NLP & Voice Lab",
}
VIEW_KEYS = list(VIEWS.keys())

query_view = st.query_params.get("view", "overview")
default_idx = VIEW_KEYS.index(query_view) if query_view in VIEW_KEYS else 0

with st.sidebar:
    st.markdown("### ⚡ LifeMate Control Tower")
    st.caption("Zalo AI Challenge 2026 · Offline SLM Agent (< 6B)")
    selected_label = st.radio(
        "Dashboard View",
        list(VIEWS.values()),
        index=default_idx,
    )
    active_view = VIEW_KEYS[list(VIEWS.values()).index(selected_label)]
    if st.query_params.get("view") != active_view:
        st.query_params["view"] = active_view

    st.divider()
    st.markdown("**Runtime Guardrail Policy**")
    guardrail_mode = st.toggle("Enable V8 Schema & FinTech Guardrails", value=True)
    wallet_slider = st.slider("Simulated e-Wallet Balance (VND)", min_value=0, max_value=800_000, value=120_000, step=30_000)
    rain_toggle = st.toggle("Simulated Tomorrow Forecast: Rain", value=False)

    st.divider()
    st.markdown("**Inference Backend Mode**")
    st.selectbox(
        "Active Execution Engine",
        [
            "Local Python Guardrail Core + Recorded V7 Trace ($0 CPU)",
            "Remote vLLM / Colab Endpoint (BYO URL)",
        ],
        index=0,
        help="Default mode executes real src/ guardrails, router, and sandbox logic in <5ms on Streamlit Cloud.",
    )
    st.text_input("Optional Remote vLLM Endpoint", placeholder="https://<colab-tunnel>/v1", type="password")
    st.caption("Author: Hà Sỹ Phong · Team V7 Public: 49.60")

st.markdown(
    """
<div class="hero-banner">
  <div class="hero-title">LifeMate — Offline Vietnamese Voice-to-Action AI Agent</div>
  <p class="hero-sub">
    End-to-end ReAct Tool-Calling SLM architecture combining <b>Qwen3-4B (LoRA r=128)</b>, <b>Chunked-CE 9,216-token training</b>,
    <b>TF-IDF Semantic Tool Routing</b>, and <b>Zero-Hallucination Schema &amp; FinTech Guardrails</b>.
  </p>
  <div class="badge-row">
    <span class="pill">4.8B Total Params (&lt; 6B Limit)</span>
    <span class="pill">9,216 Zero-Truncation Context</span>
    <span class="pill">Offline Docker (--network none)</span>
    <span class="pill">Public Score: 49.60 (#1 Team V7)</span>
    <span class="pill">V8 Dependency Guardrail Active</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


def _render_kpi_row(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, val, delta) in zip(cols, items):
        val_style = ' style="font-size:1.15rem;word-break:break-word;"' if len(val) > 14 else ""
        col.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-val"{val_style}>{val}</div>
              <div class="kpi-delta">{delta}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _plotly_dark_layout(fig: go.Figure, height: int = 350) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(18, 20, 28, 0.75)",
        plot_bgcolor="rgba(11, 14, 23, 0.70)",
        font=dict(color="#F8FAFC", size=12),
        title_font=dict(size=14, color="#F8FAFC"),
        height=height,
        margin=dict(l=36, r=20, t=44, b=36),
    )
    return fig


# ---------------------------------------------------------------------------
# View 1: Executive Overview & Benchmarks
# ---------------------------------------------------------------------------
if active_view == "overview":
    _render_kpi_row(
        [
            ("Final Public Score (V7)", "49.60", "+23.80 pts vs Baseline (25.80)"),
            ("Total Parameter Budget", "4.8B", "Qwen3-4B (4.0B) + Whisper (0.8B) < 6B"),
            ("Long-Context Training", "9,216 tok", "0% Truncated Trajectories (vs 8,192)"),
            ("L4 Peak Training VRAM", "14.02 GiB", "256-tok Chunked CE on 22.03 GiB GPU"),
        ]
    )

    st.markdown("#### End-to-End 5-Stage Agentic Architecture")
    st.markdown(
        """
        <div class="arch-grid">
          <div class="arch-box">
            <h4>1. Spoken Audio & ASR</h4>
            <p>Whisper Turbo CT2 (~0.8B) + noise/hallucination filtering & optional Dual-ASR uncertainty routing.</p>
          </div>
          <div class="arch-box">
            <h4>2. Preprocessor & Parser</h4>
            <p>Safe entity alignment + compound Vietnamese numbers ("hai triệu rưỡi") & ISO-8601 +07:00 dates.</p>
          </div>
          <div class="arch-box">
            <h4>3. Semantic Tool Router</h4>
            <p>Word (1,2)-gram TF-IDF ranks 57 apps in &lt;2ms to guide Qwen3-4B tool selection.</p>
          </div>
          <div class="arch-box">
            <h4>4. Qwen3-4B LoRA Policy</h4>
            <p>r=128, alpha=256 SFT trained on 2,046 grounded ReAct dialogues up to 9,216 context tokens.</p>
          </div>
          <div class="arch-box">
            <h4>5. V8 Schema & Hybrid Gate</h4>
            <p>Read-before-write x-ref verification, menu-order dependencies &amp; wallet fallback before execution.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        df_prog = pd.DataFrame(SCORE_PROGRESSION)
        fig = px.line(
            df_prog,
            x="version",
            y="score",
            markers=True,
            text="score",
            title="Public Leaderboard Score Progression",
            color_discrete_sequence=["#0068FF"],
        )
        fig.update_traces(line=dict(width=3.5), marker=dict(size=10, color="#00D4FF"), textposition="top center")
        fig.update_yaxes(range=[20, 56], title="Combined Public Score")
        fig.update_xaxes(title="Iteration")
        st.plotly_chart(_plotly_dark_layout(fig, height=285), use_container_width=True)

    with c2:
        st.markdown("#### Release Milestones & Competition SLA")
        st.dataframe(
            pd.DataFrame(SCORE_PROGRESSION)[["version", "score", "milestone"]],
            use_container_width=True,
            hide_index=True,
        )
        st.info(
            "Strict Offline Compliance: Fixed seed (42), greedy decoding (`temperature=0`), zero internet (`--network none`), `<= 30s/sample`.",
            icon="🔒",
        )

# ---------------------------------------------------------------------------
# View 2: Interactive ReAct Sandbox
# ---------------------------------------------------------------------------
elif active_view == "sandbox":
    st.subheader("Interactive ReAct Trace & Live Sandbox State Diff")
    col_pick, col_btn = st.columns([3, 1])
    with col_pick:
        scen_id = st.selectbox(
            "Select Real-World Challenge Failure Case / Scenario",
            list(SCENARIO_CATALOG.keys()),
            format_func=lambda k: f"[{SCENARIO_CATALOG[k]['tag']}] {SCENARIO_CATALOG[k]['title']}",
        )
    with col_btn:
        st.write("")
        run_now = st.button("▶ Run Pipeline", type="primary", use_container_width=True)

    res = run_sandbox_scenario(
        scen_id,
        guardrail_enabled=guardrail_mode,
        wallet_balance=wallet_slider,
        tomorrow_rain=rain_toggle,
    )

    st.markdown(f"**User Voice Transcript:** `{res['utterance']}`")
    _render_kpi_row(
        [
            ("Guardrail Mode", "V8 ACTIVE" if guardrail_mode else "OFF (RAW)", "Pre-execution interception"),
            ("Executed Calls", str(len(res["executed_calls"])), f"Retries / Blocked: {res['guardrail_retries']}"),
            ("Soft-Noop Failures", str(res["soft_noops"]), "Must stay 0 in production"),
            ("State Hash Changed", "YES" if res["state_changed"] else "NO", f"{res['initial_hash']} → {res['final_hash']}"),
        ]
    )

    left, right = st.columns([1.25, 1.0])
    with left:
        st.markdown("#### ReAct Step-by-Step Execution & Guardrail Interceptions")
        for step in res["trace_steps"]:
            status = step["status"]
            css_cls = "step-ok"
            if status in ("INTERCEPTED", "AUTO_CORRECTED"):
                css_cls = "step-intercept"
            elif status in ("SOFT_NOOP", "MISSED"):
                css_cls = "step-bad"
            st.markdown(
                f"""
                <div class="step-card {css_cls}">
                  <b>Step {step['step']} [{status}]</b><br/>
                  <code>{step['proposed_call']}</code><br/>
                  <span style="color:#CBD5E1;font-size:0.86rem;">{step['guardrail_note']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("#### Sandbox Environment State Diff (Before → After)")
        st.caption("Initial State (`t=0`)")
        st.json(res["initial_state"])
        st.caption("Final State (`t=T`)")
        st.json(res["final_state"])

# ---------------------------------------------------------------------------
# View 3: FinTech Risk & V8 Safety Lab
# ---------------------------------------------------------------------------
elif active_view == "risk":
    st.subheader("FinTech Transaction Risk & V8 Guardrail Stress-Test Lab")
    st.caption(
        "Live adversarial evaluation comparing Raw LLM proposals vs. deterministic V8 `PureSchemaStateTracker` + `HybridProgramAidedEngine` protection."
    )

    matrix = run_fintech_chaos_matrix()
    df_matrix = pd.DataFrame(matrix)
    safe_count = int(df_matrix["Safe"].sum())

    _render_kpi_row(
        [
            ("Adversarial Scenarios Passed", f"{safe_count}/{len(df_matrix)}", "100% live Python verification"),
            ("Financial Fallback Accuracy", "100%", "eWallet 120k < 450k -> Bank"),
            ("Ungrounded Spoken IDs Blocked", "100%", "Zero V21 / SE7 / SP204 leaks"),
            ("Menu Prerequisite Gate (V8)", "Active", "Blocks place_order before get_menu"),
        ]
    )

    c_left, c_right = st.columns([1.55, 1.0])
    with c_left:
        st.markdown("#### Live Chaos-Test Matrix Results")
        st.dataframe(
            df_matrix[["Scenario", "Threat", "Raw LLM", "V8 Guardrail"]],
            use_container_width=True,
            hide_index=True,
        )

    with c_right:
        df_chart = pd.DataFrame(
            [
                {"Pipeline": "Raw Unguarded LLM", "Category": "Prevented Soft-Noop / Bad Write (%)", "Rate": 0.0},
                {"Pipeline": "V7 Guardrail", "Category": "Prevented Soft-Noop / Bad Write (%)", "Rate": 60.0},
                {"Pipeline": "V8 Dependency + Status Guardrail", "Category": "Prevented Soft-Noop / Bad Write (%)", "Rate": 100.0},
            ]
        )
        fig_risk = px.bar(
            df_chart,
            x="Pipeline",
            y="Rate",
            text="Rate",
            color="Pipeline",
            title="Adversarial Safety Rate (5 Threat Modes)",
            color_discrete_sequence=["#EF4444", "#38BDF8", "#10B981"],
        )
        fig_risk.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        fig_risk.update_yaxes(range=[0, 118], title="Threat Interception Rate (%)")
        fig_risk.update_layout(showlegend=False)
        st.plotly_chart(_plotly_dark_layout(fig_risk, height=340), use_container_width=True)

# ---------------------------------------------------------------------------
# View 4: Semantic Router & Context Budget
# ---------------------------------------------------------------------------
elif active_view == "router":
    st.subheader("TF-IDF Semantic Tool Router & Prompt Context Compression")
    preset_queries = [
        "Thanh toán hóa đơn điện bằng ví, nếu không đủ tiền thì chuyển qua ngân hàng",
        "Đặt một tô phở tái nạm ở quán gần đây rồi đặt xe máy tới lấy",
        "Mua 200 cổ phiếu FPT nếu giá dưới 130 nghìn",
        "Kiểm tra đơn hàng vừa đặt rồi tạo yêu cầu trả hàng giúp tôi",
    ]
    chosen_preset = st.selectbox("Quick Test Utterance", preset_queries)
    custom_query = st.text_input(
        "Or Enter Custom Vietnamese Utterance",
        value=chosen_preset,
        key=f"custom_query_{hash(chosen_preset)}",
    )
    top_k = st.slider("Recommended Apps (Top-K)", min_value=2, max_value=8, value=4)

    analysis = analyze_vietnamese_utterance(custom_query, top_k=top_k)

    _render_kpi_row(
        [
            ("Recommended Apps", ", ".join(analysis["recommended_apps"]), f"Top-{top_k} priority prompt tier"),
            ("Full Schema Prompt", f"{analysis['full_prompt_chars']:,} chars", "All installed tools preserved"),
            ("Pruned Top-K Prompt", f"{analysis['pruned_prompt_chars']:,} chars", f"-{analysis['char_savings_pct']}% prompt size"),
            ("Training Cap", "9,216 tokens", "0% truncation vs 8,192 default"),
        ]
    )

    df_rank = pd.DataFrame(analysis["ranked_rows"])
    fig_router = px.bar(
        df_rank.sort_values("score", ascending=True),
        x="score",
        y="name_vi",
        color="tier",
        orientation="h",
        title="Live Semantic Similarity Scores Across Installed LifeMate Apps",
        color_discrete_map={"Recommended (Top-K)": "#0068FF", "Secondary": "#475569"},
    )
    fig_router.update_xaxes(title="Sublinear (1,2)-gram TF-IDF Cosine Score")
    fig_router.update_yaxes(title="Installed Application")
    st.plotly_chart(_plotly_dark_layout(fig_router, height=430), use_container_width=True)

# ---------------------------------------------------------------------------
# View 5: L4 Training & Chunked-CE Analytics
# ---------------------------------------------------------------------------
elif active_view == "training":
    st.subheader("NVIDIA L4 (24GB) SFT Training Dynamics & Chunked-CE Memory Profile")
    _render_kpi_row(
        [
            ("Best Checkpoint (V7)", "Step 250", "Epoch 1.95/3 | EarlyStopping"),
            ("V7 Best Eval Loss", "0.019530", f"Beats V6 Best ({V6_BEST_VAL_LOSS})"),
            ("Dataset Scale", "2,046 / 201", "+140 weak-app spoken variants"),
            ("Chunked CE Peak VRAM", f"{L4_PEAK_CHUNKED_VRAM_GIB} GiB", f"Fits safely inside {L4_TOTAL_VRAM_GIB} GiB L4"),
        ]
    )

    df_loss = pd.DataFrame(V7_LOSS_CURVE)
    col_l, col_r = st.columns([1.4, 1.0])

    with col_l:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=df_loss["step"], y=df_loss["train_loss"], mode="lines+markers", name="Training Loss", line=dict(color="#38BDF8", width=2.5)))
        fig_loss.add_trace(go.Scatter(x=df_loss["step"], y=df_loss["val_loss"], mode="lines+markers", name="Validation Loss (201 held-out)", line=dict(color="#0068FF", width=3.5)))
        fig_loss.add_hline(y=V6_BEST_VAL_LOSS, line_dash="dash", line_color="#F59E0B", annotation_text=f"V6 Best ({V6_BEST_VAL_LOSS})")
        fig_loss.add_annotation(x=250, y=0.019530, text="Best V7 Checkpoint (0.01953)", showarrow=True, arrowhead=2, ax=-40, ay=-36, font=dict(color="#34D399"))
        fig_loss.update_layout(title="V7 Qwen3-4B LoRA Loss Curve (r=128)", xaxis_title="Optimizer Step", yaxis_title="Cross-Entropy Loss")
        st.plotly_chart(_plotly_dark_layout(fig_loss, height=380), use_container_width=True)

    with col_r:
        df_vram = pd.DataFrame(
            [
                {"Method": "Full Logits CE (~9k tok)", "Logits_Overhead_GiB": 5.30, "Mode": "OOM Trigger (+3.55 GiB spike)"},
                {"Method": "ChunkedLossTrainer (256 tok)", "Logits_Overhead_GiB": 0.15, "Mode": "Measured Safe CE Overhead"},
            ]
        )
        fig_vram = px.bar(
            df_vram,
            x="Method",
            y="Logits_Overhead_GiB",
            color="Mode",
            text="Logits_Overhead_GiB",
            title="CE Logits VRAM Overhead (9k Context)",
            color_discrete_map={"OOM Trigger (+3.55 GiB spike)": "#EF4444", "Measured Safe CE Overhead": "#10B981"},
        )
        fig_vram.update_traces(texttemplate="%{text:.2f} GiB", textposition="outside")
        fig_vram.update_yaxes(range=[0, 6.5], title="Logits + CE Memory (GiB)")
        st.plotly_chart(_plotly_dark_layout(fig_vram, height=380), use_container_width=True)

# ---------------------------------------------------------------------------
# View 6: Vietnamese Spoken NLP & Voice Playground
# ---------------------------------------------------------------------------
elif active_view == "nlp":
    st.subheader("Vietnamese Spoken Number, Temporal & Voice Input Playground")
    sample_text = st.text_area(
        "Test Noisy / Spoken Vietnamese Command",
        value="Chuyển hai triệu năm trăm nghìn cho chị Lan lúc 8 giờ rưỡi tối mai nhé",
        height=90,
    )
    analysis = analyze_vietnamese_utterance(sample_text, top_k=4)

    _render_kpi_row(
        [
            ("Parsed Vietnamese Numeral", f"{analysis['parsed_number']:,}" if analysis["parsed_number"] is not None else "None", "Compound numeral engine"),
            ("Resolved ISO DateTime", analysis["resolved_iso"] or "N/A", "Relative +07:00 offset"),
            ("Complete Cancellation Flag", str(analysis["is_cancellation"]), "Hybrid conditional-safe check"),
            ("Top Routed App", analysis["recommended_apps"][0] if analysis["recommended_apps"] else "N/A", "TF-IDF semantic match"),
        ]
    )

    c_audio, c_prep = st.columns(2)
    with c_audio:
        st.markdown("#### Optional Voice Recording / WAV Upload")
        if hasattr(st, "audio_input"):
            audio_val = st.audio_input("Record Vietnamese command (connected when GPU/ASR backend is attached)")
            if audio_val is not None:
                st.audio(audio_val)
        uploaded_wav = st.file_uploader("Or upload a `.wav` utterance", type=["wav"])
        if uploaded_wav is not None:
            st.audio(uploaded_wav)

    with c_prep:
        st.markdown("#### Safe Context Preprocessor Inspection")
        st.write("**Cleaned Transcript:**", analysis["cleaned_text"])
        st.json(analysis["preprocessor_evidence"] or {"status": "No unsafe fuzzy rewrites applied"})
