# Post-Competition V8 Research & Portfolio Roadmap

Following the Zalo AI Challenge 2026 final submission (**V7: 49.60 Public Score**), this roadmap outlines concrete, measurable engineering extensions designed to push reliability, financial safety, and task completion accuracy further.

## 1. Held-Out Local Evaluation Harness (No Leaderboard Dependency)
- Split labeled development samples cleanly (`0` train/eval leakage) and score with `evaluate.py` across:
  - **Combined Score** (`0.5 * StateHashMatch + 0.5 * ToolCallF1`)
  - **Tool Error Breakdown**: `SOFT_NOOP` rate, `WRITE_BUT_STATE_UNCHANGED` rate, `REJECTED_CALLS` rate
  - **Financial & Irreversible Risk Rate**: Incorrect write calls on `banking`, `ewallet`, `stocks`, `bill_payment`, `ecommerce`, and `flights`
  - **Pareto Latency (p50 / p95 seconds/sample)** under single-GPU budget (`<= 30s/sample`)

## 2. Structural Dependency-Graph Guardrails (`src/guardrail.py`)
- Observed failure mode in V7 inference traces: multi-step apps where the model jumped directly to `place_order` before querying `get_menu` (or called `request_return` on ineligible order statuses), producing `ok=True, done=False` (`SOFT_NOOP`).
- **Extension**: Extend `PureSchemaStateTracker` with prerequisite-tool edges inferred from catalog parameter structures (`items[].item_id`, `menu`-returning read tools) and write-eligibility status checks (`cancelled` / already-returned).

## 3. Adaptive Test-Time Compute & Dual-ASR Fallback (`run_infer_best_of_n.py`, `src/asr_fallback.py`)
- Both `src/speculative_rollout.py` (`evaluate_trajectory_quality`) and `src/asr_fallback.py` (`UncertaintyAwareASRFallback` using `Qwen/Qwen3-ASR-0.6B`) exist in the codebase (`Qwen3-4B` + `Whisper-Turbo` + `Qwen3-ASR-0.6B` = `~5.4B < 6B` budget), but V7 ran single-pass greedy Whisper + Qwen3-4B.
- **Extension**: Trigger N=3 simulator speculative rollouts and Dual-ASR conflict verification **selectively** (only when pass-1 incurs `SOFT_NOOP` / `VALIDATION_RETRIES > 0` or low acoustic slot confidence on numbers/tickers/dates), targeting higher accuracy while keeping mean latency near greedy decoding.

## 4. Closed-Loop RLVR / DPO Alignment (`train_grpo_rlvr.py`)
- V7 (`checkpoint-250`, `eval_loss = 0.019530`) reached the SFT imitation ceiling via cross-entropy alone.
- **Extension**: Start from the V7 LoRA checkpoint, generate multi-turn rollouts against `zaic_tools.ToolEnv`, and optimize via Verifier-Grounded GRPO / DPO using outcome + safety rewards (`0.5 * StateMatch + 0.5 * CallF1 - SoftNoopPenalty - UnsafeFinancialWritePenalty`).

## 5. Interactive Showcase App
- Build a lightweight web UI (Gradio/Streamlit) visualizing live Vietnamese speech transcription, TF-IDF app routing, `<think>` + `<tool_call>` trace, guardrail interceptions, and before/after sandbox JSON state diffs for portfolio demonstration.
