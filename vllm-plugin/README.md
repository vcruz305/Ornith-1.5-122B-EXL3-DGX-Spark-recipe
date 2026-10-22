# vLLM route (`vllm==0.32.0` + `vcruz305/vllm-exl3`)

Serve [`vcruz305/Ornith-1.5-122B-EXL3-SAGE-2.58bpw`](https://huggingface.co/vcruz305/Ornith-1.5-122B-EXL3-SAGE-2.58bpw)
(`Qwen3_5MoeForConditionalGeneration` + `Qwen3_5MoeMTP`) on one NVIDIA DGX Spark (`GB10`, 128 GB)
through `vllm==0.32.0` and [`vcruz305/vllm-exl3`](https://github.com/vcruz305/vllm-exl3) (`main` `>= 3275819`, compiled with `-DP2B_CB=2` for `mul1`).

> **Non-destructive pack preparation by default:** `vllm-plugin/prepare_pack.sh` creates a
> symlinked view at `${PACK_DIR}-vllm` by default (`IN_PLACE=0`), so your downloaded pack
> directory is never modified in place.

All commands run from the repository root. Scripts in this folder:
- `setup_env.sh` — installs `vllm==0.32.0`, installs `vcruz305/vllm-exl3` with `NVCC_APPEND_FLAGS="-DP2B_CB=2"`, applies `patch_vllm_qwen3_5_vision_split.py`, and runs `preflight.py`.
- `prepare_pack.sh` — builds a non-destructive `-vllm` symlinked view (or prepares in-place with `IN_PLACE=1`) using `qwen_pack_scan.py`, `qwen_pack_config.py`, and `regenerate_safetensors_index.py`.
- `patch_vllm_qwen3_5_vision_split.py` — idempotent patch for `vllm/model_executor/models/qwen3_vl.py` (`Qwen3_VisionTransformer`) so split EXL3 vision `.attn.{q,k,v}_proj.*` tensors are ignored alongside the fused BF16 `attn.qkv`.
- `preflight.py` — < 5 second environment and pack validator.
- `serve.sh` — launches `vllm serve` with the optimal flags for `Ornith-1.5-122B-EXL3-SAGE-2.58bpw`.
- `probe_greedy.py` — greedy decode and `/metrics` acceptance probe.

## Quick start

### 1. Create the vLLM environment

```bash
bash vllm-plugin/setup_env.sh
```

To reuse a local checkout of `vcruz305/vllm-exl3` :

```bash
bash vllm-plugin/setup_env.sh \
  --venv ~/venvs/vllm-exl3 \
  --plugin-src /path/to/vllm-exl3 \
```

**Why `NVCC_APPEND_FLAGS="-DP2B_CB=2"` is required:** `Ornith-1.5-122B-EXL3-SAGE-2.58bpw` uses the
`mul1` codebook (`0x83DCD12D`, `.mul1` tensor markers). `vllm-exl3`'s native C++/CUDA MoE kernels
select the codebook at compile time (`P2B_CB=2` for `mul1`). `setup_env.sh` exports
`NVCC_APPEND_FLAGS="-DP2B_CB=2"` automatically.

### 2. Download the pack

```bash
hf download vcruz305/Ornith-1.5-122B-EXL3-SAGE-2.58bpw \
  --local-dir ~/models/Ornith-1.5-122B-EXL3-SAGE-2.58bpw
```

### 3. Prepare the pack view for vLLM

```bash
PACK_DIR=~/models/Ornith-1.5-122B-EXL3-SAGE-2.58bpw \
  PLUGIN_REPO=/path/to/vllm-exl3 \
  bash vllm-plugin/prepare_pack.sh
```

By default, this creates `~/models/Ornith-1.5-122B-EXL3-SAGE-2.58bpw-vllm` with symlinks to the
6 safetensors shards and tokenizer files, plus a prepared `config.json` (`codebook: "mul1"`,
`head_bits: 8`, `non_routed_exl3` including `mtp.fc`) and regenerated `model.safetensors.index.json`.
To rewrite `PACK_DIR` in place instead, pass `IN_PLACE=1` (or `--in-place`).

### 4. Run preflight checks

```bash
~/venvs/vllm-exl3/bin/python vllm-plugin/preflight.py \
  --pack ~/models/Ornith-1.5-122B-EXL3-SAGE-2.58bpw-vllm
```

### 5. Serve

```bash
source ~/venvs/vllm-exl3/bin/activate
bash vllm-plugin/serve.sh
```

`serve.sh` defaults:
- `MODEL_DIR`: auto-detected from `~/models/Ornith-1.5-122B-EXL3-SAGE-2.58bpw-vllm` or `~/models/Ornith-1.5-122B-EXL3-SAGE-2.58bpw`
- `SERVED_NAME=Ornith-1.5-122B-EXL3`
- `--quantization exl3`
- `--max-model-len 262144`
- `--gpu-memory-utilization 0.80`
- `--max-num-seqs 8`
- `--mamba-ssm-cache-dtype bfloat16`
- `--enable-prefix-caching`
- `--reasoning-parser qwen3`
- `--enable-auto-tool-choice --tool-call-parser qwen3_xml`
- `VLLM_EXL3_COOP=1` (enables fused cooperative MoE (`exl3_moe_coop`) across all 48 MoE layers + `mtp.layers.0` with pre-allocated `_COOP_TEMP_CACHE`)
- `VLLM_EXL3_COALESCE_SHARDS=1` (coalesces consecutive checkpoint shards such as `in_proj_qkv` `[3072 -> 12288]` inside `in_proj_qkvz`, saving 72 GEMV launches and 72 input Hadamard rotations per step)
- `LANGUAGE_MODEL_ONLY=1` (`--language-model-only` by default for text-only serving; setting `LANGUAGE_MODEL_ONLY=0` automatically caps `--limit-mm-per-prompt '{"image":1,"video":0}' --max-num-batched-tokens 4096` so multimodal profiling fits comfortably in 128 GB unified memory)

#### Enable multimodal vision (`LANGUAGE_MODEL_ONLY=0`)

To load the 27-layer `Qwen3_VisionTransformer` (`model.visual.*`) for image/video inputs:

```bash
LANGUAGE_MODEL_ONLY=0 bash vllm-plugin/serve.sh
```

#### Enable MTP speculative decoding (`SPEC_CONFIG`)

To enable Multi-Token Prediction (`Qwen3_5MoeMTP`) speculative decoding:

```bash
SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":1}' bash vllm-plugin/serve.sh
```

## Hardware, model, and memory

```text
GPU:          NVIDIA GB10 (DGX Spark, sm_121, aarch64)
Memory:       128 GB unified memory (121.7 GiB visible)
Engine:       vLLM 0.32.0 --quantization exl3
Model:        vcruz305/Ornith-1.5-122B-EXL3-SAGE-2.58bpw (Qwen3_5MoeForConditionalGeneration)
Weights:      41.00 GiB LM weights / 42.01 GiB total across 6 shards (no PLE table)
Context:      262,144 tokens (12 full-attention layers, 2 KV heads x 256 head_dim -> 24 KiB/token BF16 KV)
```

Because `Ornith-1.5-122B-EXL3-SAGE-2.58bpw` occupies ~41 GiB of language model weights and has no
n-gram PLE table, `GPU_MEM_UTIL=0.80` leaves over 50 GiB for the KV cache pool (~2.2M tokens of
BF16 KV capacity, or >8x concurrency at the full 262,144 context window) while retaining >20 GiB
of `MemAvailable` headroom on a single DGX Spark.

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `ValueError: There is no module or parameter named 'blocks.0.attn.k_proj'` | `LANGUAGE_MODEL_ONLY=0` without the vision split patch; run `python3 vllm-plugin/patch_vllm_qwen3_5_vision_split.py <site-packages/vllm>` |
| `error: ... does not look like a prepared vLLM pack` | Run `bash vllm-plugin/prepare_pack.sh` to create the `-vllm` view with `non_routed_exl3` and regenerated index |
| Garbage output on routed MoE layers | `vllm-exl3` was compiled without `-DP2B_CB=2` (`mul1` codebook) or `config.json` has `codebook: "mcg"`; reinstall with `NVCC_APPEND_FLAGS="-DP2B_CB=2"` and re-run `prepare_pack.sh` |
| `ValueError: There is no module or parameter named 'fc.weight'` (with MTP enabled) | `mtp.fc` missing from `non_routed_exl3` in `config.json`; re-run `vllm-plugin/prepare_pack.sh` |
