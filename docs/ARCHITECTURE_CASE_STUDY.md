# LifeMate Architecture Case Study — Offline Vietnamese Voice-to-Action Agent (`< 6B`)

**Author:** Hà Sỹ Phong (FTU HCMC) · **Codebase:** [`lifemate-vietnamese-voice-agent`](https://github.com/Phongkarazz/lifemate-vietnamese-voice-agent)

---

## 1. Problem & Offline `< 6B` System Constraints

Enterprise voice assistants executing financial transfers, medical appointments, bookings, and commerce actions cannot rely solely on cloud LLMs when strict data-residency or offline SLA constraints apply. Under the **Zalo AI Challenge 2026 (LifeMate Track)** specification, the entire speech-to-action pipeline had to operate inside an isolated GPU Docker container under hard bounds:

- **Total System Parameter Budget (`< 6B`, LLM `< 5B`):** `Qwen/Qwen3-4B` (~4.0B) + `faster-whisper-large-v3-turbo-ct2` (~0.8B) = **~4.8B parameters**.
- **Strict Offline Determinism:** `--network none`, fixed seeds, `temperature = 0.0`, `<= 16,384` context tokens, and verifiable deterministic environment transitions (`state_hash` + exact `tool_calls`).

Running `< 6B` models in multi-step Vietnamese tool calling exposes three sharp failure modes:

1. **Context-Window Truncation:** Full multi-turn ReAct tool trajectories routinely reach 8.5k–8.8k tokens, so standard 4k/8k SFT caps silently drop final `<tool_call>` supervision spans.
2. **Unsafe Hallucinated Writes (`SOFT_NOOP`):** Small models fabricate plausible-looking IDs from spoken utterances, mix up foreign keys with primary booking IDs, violate date/capacity bounds, or call checkout/order tools without reading prerequisites.
3. **Zero-Shot & Long-Horizon Fragility:** Hardcoded app rules break on unseen API domains or multi-turn dialogues exceeding ~15–20 turns.

---

## 2. Zero-Truncation 9,216-Token L4 Fine-Tuning (`ChunkedLossTrainer`)

Fine-tuning `Qwen/Qwen3-4B` (`bf16`, `LoRA r=128, alpha=256`, `paged_adamw_8bit`) up to `MAX_SEQ_LEN = 9216` on a single **24GB NVIDIA L4 (`22.03 GiB` usable VRAM)** naively allocates `[B, T, V]` float32 logits tensors (`~5.30 GiB` for the cross-entropy head alone), triggering out-of-memory failures.

- **Engineering Fix:** Implemented **`ChunkedLossTrainer`** inside [`train_sft.py`](../train_sft.py) and [`notebooks/Train_Qwen3_4B_ZAIC2026_V7_L4_Max.ipynb`](../notebooks/Train_Qwen3_4B_ZAIC2026_V7_L4_Max.ipynb), computing `lm_head` projections and cross-entropy strictly over active assistant target slices in **256-token chunks**.
- **Measured Result:** Reduced cross-entropy logits peak memory from **5.30 GiB -> 0.15 GiB**, held peak L4 training VRAM at **14.02 / 22.03 GiB**, and improved validation loss cleanly from **0.020576 (V6)** down to **0.019530 (V7, `checkpoint-250`)**.

---

## 3. Competition Results & Production Debugging Post-Mortem

### Leaderboard Progression & Root-Cause Recovery
- **Public Leaderboard Progression:** `25.80` → `34.40` → `44.20` → **`49.30` (V6)** → **`49.60` (V7 Final)**
- **Private Evaluation Recovery:** **`26.40`** → **`36.05` (`+9.65` points)**

**What caused the initial `26.40` score, and how was it resolved?**
1. **Silent ASR Checkpoint Mismatch:** Auditing the first private inference run revealed a silent fallback: the notebook configuration pointed to the base `deepdml/faster-whisper-large-v3-turbo-ct2` model (`ASR_LORA = None`) instead of the fine-tuned Vietnamese Whisper CT2 checkpoint. Phonetic slips on Vietnamese brand names, food items, medical specialties, and spoken numerals propagated downstream into empty read searches (`done=True, results=[]`) and stalled agent loops.
2. **Cache/Resume Collision Hazard:** Re-running directly against shared output paths risked reusing base-ASR transcript caches or `--resume` checkpoints from the defective run. Isolating both the ASR JSON cache path and output JSONL path forced clean end-to-end re-transcription with the fine-tuned Vietnamese checkpoint.
3. **Deterministic Audit Verification:** Before resubmitting, every generated output was re-verified for 100% ID ordering, schema validity, zero `ToolEnv` invalid calls, and zero `state_hash` replay mismatches, raising the Private Score directly from **`26.40` to `36.05` (`+9.65` points)**.

### General Post-Competition Engineering Lessons (Motivating V9)
Evaluating distribution shifts across complex multi-turn tasks surfaced three universal architectural bottlenecks in `< 6B` offline agents:
- **Long-Horizon Context Bloat & Lost-in-the-Middle:** Retaining unpruned reviews, descriptions, and metadata from early tool calls inflates prompts across 15–24+ turn conversations, obscuring critical entity IDs from earlier turns.
- **Multi-Turn ASR Error Compounding:** Phonetically similar proper nouns accumulate across multi-turn spoken dialogues when ASR initial prompts rely solely on static dictionaries.
- **Hardcoded Assumptions vs. Unseen Tool Schemas:** Exact singular/plural collection string checks (e.g., `booking` returned by a create tool vs `bookings` declared in `x-ref`) and static keyword lists fail when unseen API domains or foreign keys appear at test time.

---

## 4. V9 Schema-First Architecture (`src/`)

To solve those bottlenecks generically without per-app hardcoding, the codebase was extended post-competition into **V9 Schema-First**:

1. **V9 Schema-First Guardrails ([`src/guardrail.py`](../src/guardrail.py)):**
   - **Singular/Plural & Generic Collection Normalization:** `_normalize_collection_key` and `_collections_match` reconcile `app.booking` with `app.bookings` (plus generic keys like `item` / `entry`) within the same application, eliminating false-positive blocks on reused history entities while preserving strict cross-app boundaries (`pantry` ↔ `grocery` remains locked).
   - **Primary ID vs. Foreign Key Isolation:** `_record_entity_dict` keeps an entity's primary ID inside its collection membership set while isolating foreign keys (e.g., `hotel_id` nested inside a `booking`), blocking FK confusion at write time.
   - **Schema-Inferred Execution Gates:** Automatically enforces `x-one-of` parameter requirements, date ordering (`check_in < check_out`, `start_date <= end_date`), ISO-8601 `+07:00` timezones, guest capacity (`guests <= max_guests`), availability flags (`available=True`), `room_type` membership, `slots` / operating hours (`open..close`), and media `episode` bounds directly from schema & observed state.
2. **Zero-Shot Catalog ASR Hotwords ([`src/asr.py`](../src/asr.py)):**
   - `build_asr_hotwords_prompt` harvests `name_vi`, concise Vietnamese description clauses, readable parameter `enum` terms, and single-dict/list history entity names automatically for any installed app without code edits.
3. **History-Aware Tool Routing ([`src/router.py`](../src/router.py)):**
   - `SemanticToolRouter` combines sublinear `(1,2)`-gram TF-IDF similarity with a calibrated recent-app boost (`+0.12` / `+0.06` across recent assistant turns) so elliptic follow-ups (*"Hủy cái lúc nãy giúp anh"*) resolve accurately while explicit topic switches still override cleanly.
4. **24-Turn Token-Budgeted History Compaction ([`src/prompt_builder.py`](../src/prompt_builder.py)):**
   - `compact_tool_result` and `format_history_block` preserve the latest turns verbatim while compacting bulky prose (`reviews`, `amenities`, `description`) in earlier turns and preserving **100% of entity IDs** across 24-turn dialogues.

---

## 5. Reproducible CPU Benchmark (`scripts/benchmarks/benchmark_v9_suite.py`)

Measured locally on CPU using 100% clean synthetic scenarios (`python3 scripts/benchmarks/benchmark_v9_suite.py`):

### LifeMate V9 Reproducible CPU Benchmark (100% Synthetic Suite)

| Metric | Raw LLM | V7 Legacy Rules | V9 Schema-First |
|---|---:|---:|---:|
| Unsafe Write Interception (10 adversarial cases) | 0.0% | 20.0% | **100.0%** |
| History Entity False-Positive Block Rate (5 follow-ups) | 0.0% (no guard) | 100.0% | **0.0%** |
| 24-Turn History Prompt Savings (chars) | 0.0% (104,906) | 0.0% (104,906) | **-74.0% (27,268)** |
| 24-Turn Entity ID Retention | — | — | **100.0%** (48/48) |
| Zero-Shot Unseen App Hotword Coverage (6 apps) | 0.0% | 0.0% | **100.0%** |
| Elliptic Follow-Up Top-1 Routing (5 dialogues) | 20.0% | 20.0% | **100.0%** |
| Guardrail Pre-Execution CPU Latency (p50 / p95 over 500 calls) | 0 ms | — | **0.012 ms / 0.013 ms** |

---

## 6. Attribution & Data Compliance

- **Origin & Team Contribution:** Originated from our team's Zalo AI Challenge 2026 LifeMate repository (`sun0810pi/zalo-challenge`).
- **Personal Scope (Hà Sỹ Phong):** Designed and tuned the L4 Max `ChunkedLossTrainer` 9,216-token SFT pipelines (`49.30` V6 → `49.60` V7 team-best Public Score), diagnosed and resolved the silent Whisper checkpoint & cache/resume mismatch on Private evaluation (`26.40` → `36.05`, `+9.65` points), engineered the post-competition **V9 Schema-First** safety/compaction/routing architecture, built the reproducible CPU benchmark suite, and designed the minimalist Streamlit product demo.
- **Compliance:** Competition datasets (`train/`, `public_test/`, `private_test/`, `.wav` audio, and `.jsonl` submissions) are strictly excluded from this repository.
