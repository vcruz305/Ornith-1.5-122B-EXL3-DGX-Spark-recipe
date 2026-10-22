# AGENTS.md: instructions for AI agents setting up this recipe

Read this before running anything in this repository.

## Supported serving path: `vllm-plugin/` (`vllm==0.32.0` + `vcruz305/vllm-exl3`)

This repository serves
[`vcruz305/Ornith-1.5-122B-EXL3-SAGE-2.58bpw`](https://huggingface.co/vcruz305/Ornith-1.5-122B-EXL3-SAGE-2.58bpw)
(`Qwen3_5MoeForConditionalGeneration` + `Qwen3_5MoeMTP`, base model [`ornith-ai/Ornith-1.5-122B`](https://huggingface.co/ornith-ai/Ornith-1.5-122B))
on one NVIDIA DGX Spark (`GB10`, `sm_121`, aarch64) through `vllm==0.32.0` and
[`vcruz305/vllm-exl3`](https://github.com/vcruz305/vllm-exl3) (`main` `>= 3275819`):

```bash
bash vllm-plugin/setup_env.sh                    # installs vllm==0.32.0, vllm-exl3 (-DP2B_CB=2), and vision patch
HF_XET_HIGH_PERFORMANCE=1 ~/venvs/vllm-exl3/bin/hf download \
  vcruz305/Ornith-1.5-122B-EXL3-SAGE-2.58bpw \
  --local-dir ~/models/Ornith-1.5-122B-EXL3-SAGE-2.58bpw
PACK_DIR=~/models/Ornith-1.5-122B-EXL3-SAGE-2.58bpw \
  PLUGIN_REPO=/path/to/vllm-exl3 \
  bash vllm-plugin/prepare_pack.sh               # creates non-destructive symlink view at <PACK_DIR>-vllm by default
source ~/venvs/vllm-exl3/bin/activate
bash vllm-plugin/serve.sh                        # OpenAI API on http://127.0.0.1:8899/v1
```

The model ID to send in API requests is `Ornith-1.5-122B-EXL3`. Check readiness with
`curl -s http://127.0.0.1:8899/v1/models`.

By default, `vllm-plugin/prepare_pack.sh` creates a non-destructive symlinked view at
`PACK_VIEW_DIR` (`${PACK_DIR}-vllm`) so the original downloaded pack at `PACK_DIR` is never modified
in place. Set `IN_PLACE=1` only if you intentionally want to rewrite `PACK_DIR` in place.

## Do not

- **Do not build `vllm-exl3` without `NVCC_APPEND_FLAGS="-DP2B_CB=2"`.** This pack uses the
  `mul1` codebook (`0x83DCD12D`), so the native `vllm_exl3_c` kernels must be compiled with
  `-DP2B_CB=2`. `vllm-plugin/setup_env.sh` sets this automatically.
- **Do not disable `VLLM_EXL3_COOP=1` or `VLLM_EXL3_COALESCE_SHARDS=1`.** Together they deliver a
  `+33.6%` decode speedup (`22.84 -> 30.51 tok/s` no-draft, `32.60 tok/s` with `MTP k=1`) and make
  all 48 routed MoE layers 100% CUDA-graph capturable (`2.0 s` capture time).
- **Do not set `EXL3_INT8_GEMV_MAX_K=8`.** Empirical GB10 ablation showed `EXL3_INT8_GEMV_MAX_K=8`
  regresses decode (`30.51 -> 29.74 tok/s`) because `lm_head` (`[3072 -> 248320]`, `K=8`) is faster on
  `exl3_gemm_coop` (`3,146 µs` vs `3,229 µs`). Leave default `MAX_K=6`.
- **Do not raise `MAX_MODEL_LEN` / `--max-model-len` above `262144`.** That is the model's trained
  context window.

## Sizing (context, memory, and concurrency)

Ornith-1.5-122B (`Qwen3_5MoeForConditionalGeneration`) has 48 layers (36 Gated DeltaNet linear
attention layers and 12 full-attention GQA layers with `full_attention_interval: 4`, 2 KV heads
at `head_dim: 256`), 256 routed experts (top-8) + 1 shared expert per layer, 1 native MTP layer,
and a 27-layer vision tower. Ornith-1.5 has **no** n-gram PLE table, so the `41.00 GiB` language
model (`42.01 GiB` total across 6 shards) leaves over `50 GiB` for the KV cache pool at
`GPU_MEM_UTIL=0.80` on a 128 GB DGX Spark.

In `vllm-plugin/serve.sh`:
- Defaults to `LANGUAGE_MODEL_ONLY=1` (`--language-model-only`), `VLLM_EXL3_COOP=1`,
  `VLLM_EXL3_COALESCE_SHARDS=1`, `--max-model-len 262144`, `--mamba-ssm-cache-dtype bfloat16`,
  `--enable-prefix-caching`, `--reasoning-parser qwen3`, `--enable-auto-tool-choice`,
  `--tool-call-parser qwen3_xml`.
- Set `LANGUAGE_MODEL_ONLY=0` to load the 27-layer vision tower for multimodal requests (automatically
  applies `--limit-mm-per-prompt '{"image":1,"video":0}' --max-num-batched-tokens 4096` so multimodal
  profiling fits cleanly in 128 GB unified memory).
- Set `SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":1}'` to enable MTP speculative decoding
  in vLLM (`32.60 tok/s` / `32.99` peak on code).

## Verify you are on the right runtime

```bash
python3 vllm-plugin/preflight.py --pack ~/models/Ornith-1.5-122B-EXL3-SAGE-2.58bpw-vllm
```

Run CPU regression checks before deployment:

```bash
python3 -m unittest discover -s bench -p 'test_*.py' -v
bash -n vllm-plugin/setup_env.sh vllm-plugin/prepare_pack.sh vllm-plugin/serve.sh
```

Do not report requested `max_tokens` as emitted tokens. Preserve actual API usage, cache counts,
server timings, errors, and model identity.

## If something fails

- `missing .../config.json` or `does not look like a prepared vLLM pack`: run `bash vllm-plugin/prepare_pack.sh`
  to create the `${PACK_DIR}-vllm` symlinked view with `non_routed_exl3` and regenerated index.
- `ValueError: There is no module or parameter named 'blocks.0.attn.k_proj'` under vLLM with
  `LANGUAGE_MODEL_ONLY=0`: run `python3 vllm-plugin/patch_vllm_qwen3_5_vision_split.py <venv>`.
