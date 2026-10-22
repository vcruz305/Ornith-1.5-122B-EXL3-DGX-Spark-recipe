# Ornith-1.5-122B EXL3 on NVIDIA DGX Spark

Serve [`vcruz305/Ornith-1.5-122B-EXL3-SAGE-2.58bpw`](https://huggingface.co/vcruz305/Ornith-1.5-122B-EXL3-SAGE-2.58bpw)
(base model [`ornith-ai/Ornith-1.5-122B`](https://huggingface.co/ornith-ai/Ornith-1.5-122B)) on a single
NVIDIA DGX Spark (`GB10`, `sm_121`, 128 GB unified memory) with **`vllm==0.32.0`**
(`Qwen3_5MoeForConditionalGeneration` + `Qwen3_5MoeMTP`) and
[**`vcruz305/vllm-exl3`**](https://github.com/vcruz305/vllm-exl3) (`-DP2B_CB=2` for the `mul1` codebook).

SAGE is our own mixed-precision quantization method for EXL3. The method is not published. If a tensor is EXL3, its K is self-describing on the trellis.

**Start here:** [quick start (`vLLM + vllm-exl3`)](#quick-start-vllm--vllm-exl3),
[measured GB10 performance](#measured-performance-on-1-nvidia-dgx-spark-gb10-sm_121-128-gb-unified),
[9-suite downstream benchmark vs. 4.0 bpw](#2-9-suite-353356-item-downstream-benchmark-vs-mia-ailabornith-15-122b-exl3-eaqv1-4-bpw),
[fidelity vs. BF16 master](#fidelity-vs-unquantized-ornith-aiornith-15-122b-bf16-master),
[tool calling](docs/tool-calling.md),
[benchmark instructions](bench/README.md), and
[service guide](docs/service.md).
AI agents should first read [AGENTS.md](AGENTS.md).

## Model

| Field | Value |
|---|---|
| Checkpoint served | [`vcruz305/Ornith-1.5-122B-EXL3-SAGE-2.58bpw`](https://huggingface.co/vcruz305/Ornith-1.5-122B-EXL3-SAGE-2.58bpw) |
| Base model | [`ornith-ai/Ornith-1.5-122B`](https://huggingface.co/ornith-ai/Ornith-1.5-122B) |
| Architecture | `Qwen3_5MoeForConditionalGeneration` |
| Layers | 48 hidden layers (36 Gated DeltaNet linear attention + 12 full attention, `full_attention_interval: 4`) |
| MoE configuration | 256 routed experts (top-8 routing) + 1 shared expert per layer (`hidden_size: 3072`, `moe_intermediate_size: 1024`) |
| Speculative drafter | 1 native Multi-Token Prediction (`mtp.*`) layer (`0.59 GiB`) |
| Vision tower | 27-layer `Qwen3_VisionTransformer` (`model.visual.*`, `0.37 GiB` at 4-bit EXL3 + fused BF16 `attn.qkv`) |
| Context window | `262,144` tokens (`max_position_embeddings`) |
| Codebook & head | `mul1` (`0x83DCD12D`), `head_bits: 8` (`lm_head` at K=8) |
| Pack size | `41.00 GiB` language model weights (`39.58 GiB` GPU-resident LM + `1.42 GiB` `embed_tokens`) / `42.01 GiB` total across 6 safetensors shards |

Every routed MoE layer uses uniform per-layer `(gate/up, down)` bitwidths across all 256 experts (`18` trunk layers at `(3, 3, 3)` and `30` trunk layers at `(2, 2, 3)`, plus `mtp.layers.0` at `(2, 2, 2)`, with zero 2-bit trunk `down_proj` matrices) so all `48 / 48` trunk layers and `mtp.layers.0` execute through the fast fused cooperative MoE kernel (`VLLM_EXL3_COOP=1`).

## Measured Performance on 1× NVIDIA DGX Spark (`GB10`, `sm_121`, 128 GB Unified)

### 1. `vLLM 0.32.0` + `vcruz305/vllm-exl3` (`3275819`) Optimization Progression & Throughput (with `REXL3` Reference)

| Configuration / Workload | Decode (`tok/s`) | Per-Stream (`tok/s`) | Prefill (`tok/s`) | TTFT (`ms`) | Notes |
|---|---:|---:|---:|---:|---|
| **Baseline `vllm-exl3` (`VLLM_EXL3_COOP=0`, uncoalesced)** | 22.84 | 22.84 | — | 385.0 | Uncoalesced `in_proj_qkvz`; `11.0 s` CUDA graph capture |
| **`+ VLLM_EXL3_COOP=1` (`exl3_moe_coop` all 48 layers + `_COOP_TEMP_CACHE`)** | 29.40 | 29.40 | — | 111.5 | **+28.7% decode**, 100% CUDA-graph capturable (`2.0 s` capture) |
| **`+ VLLM_EXL3_COALESCE_SHARDS=1` (`in_proj_qkv` `[3072->12288]`) + `bfloat16` SSM (Default `serve.sh`)** | **30.51** *(30.57 peak)* | **30.51** | **1,119.1** *(2K)* | **109.1** | **+33.6% over baseline**, saves 72 GEMV launches + 72 Hadamards/step |
| **`+ MTP Speculative Decoding (`k=1`, Greedy `T=0.0` Code)** | **32.60** *(32.78 peak)* | **32.60** | — | 124.2 | **+42.7% over baseline** (`88.6%` draft acceptance on code) |
| **`+ MTP Speculative Decoding (`k=1`, Vendor `T=1.0, top_p=0.95, top_k=20` Code)** | **31.56** *(32.99 peak)* | **31.56** | — | 125.0 | `30.38 tok/s` mean across code, math, reasoning, and general |
| **Concurrent Decode `C = 1` (`vLLM` default `serve.sh`)** | **30.51** | **30.51** | — | 109.1 | Single active stream (`45.88 GiB` weights + `0.36 GiB` graph pool) |
| **Concurrent Decode `C = 4` (`vLLM` default `serve.sh`)** | **63.07** | **17.22** | — | 238.1 | **2.07× aggregate throughput** |
| **Concurrent Decode `C = 8` (`vLLM` default `serve.sh`)** | **89.21** | **12.02** | — | 428.7 | **2.92× aggregate throughput** |
| **Prefill `2,048` tokens (`vLLM` Cold / Prefix-Cached)** | — | — | **1,119.1 / 1,408.7** | 1,830.1 / 1,453.9 | **1.26×** cached speedup |
| **Prefill `4,096` tokens (`vLLM` Cold / Prefix-Cached)** | — | — | **1,033.8 / 2,210.9** | 3,962.1 / 1,852.6 | **2.14×** cached speedup |
| **Prefill `8,192` tokens (`vLLM` Cold / Prefix-Cached)** | — | — | **991.6 / 4,449.5** | 8,261.2 / 1,841.1 | **4.49×** cached speedup |
| *Reference: `REXL3` Native C++/CUDA Engine (`C = 1` / `C = 8`)* | *30.58 / 90.82* | *30.58 / 11.35* | *—* | *91.2* | *`9.12 s` cold boot; `vLLM + vllm-exl3` matches `REXL3` within `0.2%–1.8%` and exceeds `C=1` with `MTP k=1` (`32.60 tok/s`)* |

### 2. 9-Suite (`353`/`356`-Item) Downstream Benchmark vs. `Mia-AiLab/Ornith-1.5-122B-EXL3-EAQv1-4-bpw`

Measured on the same DGX Spark (`GB10`) under identical deterministic prompts and harnesses:

| Benchmark Suite | Items | **`vcruz305` SAGE 2.58 bpw (`41.00 GiB` LM)** | `Mia-AiLab` EAQv1 4.0 bpw (`59.00 GiB` LM) | Delta (`SAGE 2.58` vs `Mia 4.0`) |
|---|---:|---:|---:|---:|
| **MMLU-Pro (STEM)** | 100 | 94 / 100 (`94.00%`) | **96 / 100 (`96.00%`)** | -2 (`-2.00%`) |
| **LiveCodeBench (Matched / Full)** | 17 / 20 | **13 / 17 (`76.47%`) / 15 / 20 (`75.00%`)** | 12 / 17 (`70.59%`) / 14 / 20 (`70.00%`) | **+1 (`+5.88%` / `+5.00%`)** |
| **MBPP+** | 60 | **53 / 60 (`88.33%`)** | 51 / 60 (`85.00%`) | **+2 (`+3.33%`)** |
| **HumanEval+** | 40 | **39 / 40 (`97.50%`)** | **39 / 40 (`97.50%`)** | 0 (`0.00%`) |
| **SciCode** | 10 | **5 / 10 (`50.00%`)** | 4 / 10 (`40.00%`) | **+1 (`+10.00%`)** |
| **MATH-500** | 40 | 38 / 40 (`95.00%`) | **39 / 40 (`97.50%`)** | -1 (`-2.50%`) |
| **MGSM (8 Languages)** | 40 | **40 / 40 (`100.00%`)** | **40 / 40 (`100.00%`)** | 0 (`0.00%`) |
| **BFCL (Function Calling)** | 40 | **37 / 40 (`92.50%`)** | **37 / 40 (`92.50%`)** | 0 (`0.00%`) |
| **Multi-step Tool Chains** | 6 | **5 / 6 (`83.33%`)** | **5 / 6 (`83.33%`)** | 0 (`0.00%`) |
| **Coding Subtotal (LCB+MBPP+HE+SciCode)** | **130** | **112 / 130 (`86.15%`)** | 108 / 130 (`83.08%`) | **+4 (`+3.08%`)** |
| **OVERALL TOTAL (353-Item Matched)** | **353** | **324 / 353 (`91.78%`)** | 323 / 353 (`91.50%`) | **+1 (`+0.28%`)** |
| **OVERALL TOTAL (356-Item Full)** | **356** | **326 / 356 (`91.57%`)** | 325 / 356 (`91.29%`) | **+1 (`+0.28%`)** |
| **Loaded LM Weight Footprint** | — | **41.00 GiB** *(fits 1× 48 GB GPU)* | 59.00 GiB *(requires 80 GB+)* | **-18.00 GiB (`-30.5%` smaller)** |
| **Cold Boot Time (`REXL3`, `GB10`)** | — | **9.12 s** | 19.23 s | **2.11× faster** |
| **`C = 8` Aggregate Decode (`GB10`)** | — | **89.21 tok/s** *(`vLLM`)* / **90.82 tok/s** *(`REXL3`)* | 83.11 tok/s | **+9.3% faster** |

## Fidelity vs. Unquantized `ornith-ai/Ornith-1.5-122B` BF16 Master

All metrics below are measured directly against the unquantized `ornith-ai/Ornith-1.5-122B` BF16 master across three strictly 256-token block-disjoint evaluation splits (`e2e_eval`: 16,368 positions across General, Code, Math, and Reasoning; `heldout_8x1024`: 8,184 positions; `indep_8x1024`: 8,184 positions) over the full 248,320-token vocabulary:

| Evaluation Split / Subset | Positions | Top-1 Agreement | Top-5 Agreement | Mean KLD (nats) | Median KLD (nats) | Mean TV | BF16 Master PPL | SAGE 2.58bpw PPL | $\Delta$PPL (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`e2e_eval` (Answer-Only)** | 5,256 | **92.10%** | **99.24%** | **0.10845** | **0.00175** | 0.07938 | 2.2139 | 2.2984 | +3.82% |
| **`e2e_eval` (Non-Header)** | 16,120 | **91.12%** | **99.09%** | **0.16414** | **0.00383** | 0.09176 | 2.6812 | 2.7426 | **+2.29%** |
| **`e2e_eval` — Code (HumanEval)** | 4,038 | **94.43%** | **99.21%** | **0.14847** | **0.00041** | 0.06060 | 1.6854 | 1.7555 | +4.16% |
| **`e2e_eval` — Reasoning (MATH-500)** | 4,036 | **91.18%** | **99.28%** | **0.15074** | **0.00516** | 0.08967 | 3.0384 | 2.9640 | **-2.45%** |
| **`e2e_eval` — Math (GSM8K)** | 4,018 | **90.24%** | **99.08%** | **0.20349** | **0.00491** | 0.10177 | 2.3485 | 2.4310 | +3.51% |
| **`e2e_eval` — General (WikiText-2)** | 4,028 | **88.60%** | **98.81%** | **0.15403** | **0.02442** | 0.11510 | 4.2996 | 4.4758 | +4.10% |
| **`heldout_8x1024` (Answer-Only)** | 2,664 | **90.58%** | **99.29%** | **0.09522** | **0.00299** | 0.08531 | 2.4456 | 2.5333 | +3.59% |
| **`heldout_8x1024` — Code** | 2,026 | **94.82%** | **99.41%** | **0.14258** | **0.00035** | 0.05740 | 1.6215 | 1.6995 | +4.81% |
| **`heldout_8x1024` — Reasoning** | 2,024 | **91.90%** | **99.11%** | **0.13906** | **0.00182** | 0.08081 | 2.4344 | 2.4775 | +1.77% |
| **`indep_8x1024` (Non-Header)** | 8,056 | **90.83%** | **99.14%** | **0.14847** | **0.00575** | 0.09303 | 2.8513 | 2.8938 | **+1.49%** |
| **`indep_8x1024` — Code** | 2,012 | **94.43%** | **99.25%** | **0.14510** | **0.00039** | 0.05797 | 1.7349 | 1.7913 | +3.25% |
| **`indep_8x1024` — Reasoning** | 2,026 | **93.09%** | **99.51%** | **0.08431** | **0.00607** | 0.07276 | 2.7303 | 2.7202 | **-0.37%** |
| **`indep_8x1024` — Math** | 2,010 | **90.00%** | **99.05%** | **0.17364** | **0.00998** | 0.10259 | 2.9910 | 3.0177 | **+0.89%** |

## Quick start (`vLLM` + `vllm-exl3`)

Run these commands on your DGX Spark from a clone of this repository (full details in [`vllm-plugin/README.md`](vllm-plugin/README.md)):

```bash
# 1. Create the vLLM environment (vllm==0.32.0, vcruz305/vllm-exl3 with -DP2B_CB=2, vision patch).
bash vllm-plugin/setup_env.sh

# 2. Download the pack (skip if already downloaded).
HF_XET_HIGH_PERFORMANCE=1 ~/venvs/vllm-exl3/bin/hf download \
  vcruz305/Ornith-1.5-122B-EXL3-SAGE-2.58bpw \
  --local-dir ~/models/Ornith-1.5-122B-EXL3-SAGE-2.58bpw

# 3. Prepare a non-destructive symlinked view at ~/models/Ornith-1.5-122B-EXL3-SAGE-2.58bpw-vllm
#    (leaves the downloaded pack directory untouched).
PACK_DIR=~/models/Ornith-1.5-122B-EXL3-SAGE-2.58bpw \
  PLUGIN_REPO=/path/to/vllm-exl3 \
  bash vllm-plugin/prepare_pack.sh

# 4. Verify environment and prepared pack (< 5 seconds).
~/venvs/vllm-exl3/bin/python vllm-plugin/preflight.py \
  --pack ~/models/Ornith-1.5-122B-EXL3-SAGE-2.58bpw-vllm

# 5. Launch vLLM (--quantization exl3, 262,144 context, VLLM_EXL3_COOP=1, VLLM_EXL3_COALESCE_SHARDS=1).
source ~/venvs/vllm-exl3/bin/activate
bash vllm-plugin/serve.sh
```

Set `LANGUAGE_MODEL_ONLY=0 bash vllm-plugin/serve.sh` to enable the 27-layer vision tower, or
`SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":1}' bash vllm-plugin/serve.sh` to enable
MTP speculative decoding (`32.60 tok/s` / `32.99` peak on code).

The API listens at `http://127.0.0.1:8899/v1` with served model ID `Ornith-1.5-122B-EXL3`:

```bash
curl --fail-with-body http://127.0.0.1:8899/v1/models

curl --fail-with-body http://127.0.0.1:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Ornith-1.5-122B-EXL3",
    "messages": [{"role": "user", "content": "Write a short Python example."}],
    "chat_template_kwargs": {"enable_thinking": false},
    "max_tokens": 256,
    "stream": false
  }'
```

## Context, memory, and concurrency

Only 12 of the 48 layers are full attention (`full_attention_interval: 4`, 2 KV heads at `head_dim: 256`),
so BF16 KV storage costs `24 KiB/token` (`~6.0 GiB` per 262,144 tokens). Combined with the `41.00 GiB`
language model weights (and zero PLE table overhead), `GPU_MEM_UTIL=0.80` on a single 128 GB DGX Spark
leaves over `50 GiB` for the KV cache pool (`~2.2M` tokens of BF16 KV capacity, or `8+` concurrent
262,144-token requests) while retaining `>20 GiB` of `MemAvailable` headroom.

Preview the exact `vllm serve` command without launching the server:

```bash
DRY_RUN=1 bash vllm-plugin/serve.sh
```

## Tool calling

The server enables Qwen XML tool calling (`--tool-call-parser qwen3_xml --enable-auto-tool-choice`)
and `<think>` reasoning parsing (`--reasoning-parser qwen3`).
See [the tool-calling guide](docs/tool-calling.md) and [API validation guide](docs/api-validation.md):

```bash
python3 bench/tool_smoke.py \
  --model Ornith-1.5-122B-EXL3 --mode both --case all \
  --output results/tool-smoke.json
```

To run the reproducible API benchmark suite:

```bash
python3 bench/bench_v1.py \
  --model Ornith-1.5-122B-EXL3 \
  --suite all --repeat 3 --warmup 1 --run-id ornith-v1 \
  --output results/api-benchmark.json
```

## Persistent service and remote access

[The service guide](docs/service.md) provides a user-systemd unit for running `vllm-plugin/serve.sh`
bound to `127.0.0.1:8899`. Access a loopback server from another machine through an SSH tunnel:

```bash
ssh -N -L 8899:127.0.0.1:8899 YOUR_SPARK_USER@YOUR_SPARK_IP
```

## Troubleshooting

| Symptom | Action |
|---|---|
| `error: ... does not look like a prepared vLLM pack` | Run `bash vllm-plugin/prepare_pack.sh` to create the `-vllm` view with `non_routed_exl3` and regenerated index. |
| `blocks.0.attn.k_proj` error in vLLM with `LANGUAGE_MODEL_ONLY=0` | Apply [`vllm-plugin/patch_vllm_qwen3_5_vision_split.py`](vllm-plugin/patch_vllm_qwen3_5_vision_split.py). |
| Garbage output on routed MoE layers | Ensure `vcruz305/vllm-exl3` (`>= 3275819`) was built with `NVCC_APPEND_FLAGS="-DP2B_CB=2"` (`mul1` codebook) and `prepare_pack.sh` was run. |
| Model ID differs from client configuration | Query `/v1/models` and use its advertised ID (`Ornith-1.5-122B-EXL3`). |

## Repository layout

| Path | Purpose |
|---|---|
| [`vllm-plugin/`](vllm-plugin/README.md) | `vllm==0.32.0` + `vcruz305/vllm-exl3` setup, non-destructive pack preparation, preflight, memory advisory, and server launcher |
| [`HISTORICAL_BENCHMARKS.md`](HISTORICAL_BENCHMARKS.md) | Full GB10 kernel microbenchmarks, `vLLM` A/B ablation ledger, and `REXL3` reference measurements |
| [`bench/`](bench/) | Strict `/v1` API benchmarks, concurrency, long-context retrieval, resilience, and tool/SDK checks |
| [`docs/tool-calling.md`](docs/tool-calling.md) | Qwen XML tool-calling semantics and validation commands |
| [`docs/service.md`](docs/service.md) | Persistent user-systemd deployment for `vllm-plugin/serve.sh` |
| [`docs/memory.md`](docs/memory.md) | Header-based weight and KV cache memory estimation |
| [`docs/api-validation.md`](docs/api-validation.md) | Serial API resilience and automatic tool-choice validation suites |

## Credits

This recipe builds on [`ornith-ai/Ornith-1.5-122B`](https://huggingface.co/ornith-ai/Ornith-1.5-122B)
and [vllm-project/vllm](https://github.com/vllm-project/vllm), with the EXL3 serving plugin maintained in
[vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3).

## License

MIT License — see [LICENSE](LICENSE). Model weights are not redistributed here; see the model card on Hugging Face for weight licensing.
