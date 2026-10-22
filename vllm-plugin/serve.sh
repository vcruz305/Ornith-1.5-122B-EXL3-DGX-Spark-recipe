#!/usr/bin/env bash
# Serve vcruz305/Ornith-1.5-122B-EXL3-SAGE-2.58bpw (after vllm-plugin/prepare_pack.sh)
# through vLLM 0.32.0 + vcruz305/vllm-exl3 on one DGX Spark (GB10).
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-}"
if [[ -z "$MODEL_DIR" ]]; then
  for cand in \
    "$HOME/models/Ornith-1.5-122B-EXL3-SAGE-2.58bpw-vllm" \
    "$HOME/models/Ornith-1.5-122B-EXL3-SAGE-2.58bpw" \
    "$HOME/models/Ornith-1.5-122B-EXL3"; do
    [[ -f "$cand/config.json" ]] && { MODEL_DIR="$cand"; break; }
  done
  MODEL_DIR="${MODEL_DIR:-$HOME/models/Ornith-1.5-122B-EXL3-SAGE-2.58bpw-vllm}"
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8899}"
SERVED_NAME="${SERVED_NAME:-Ornith-1.5-122B-EXL3}"

# Full 262,144 context window: only 12 of 48 layers are full attention
# (full_attention_interval: 4) with 2 KV heads at head_dim 256, so KV cache
# costs 24,576 bytes/token in BF16 (~6.0 GiB per 262,144 tokens). Because
# Ornith-1.5-122B EXL3 2.58bpw weights occupy only ~41 GiB (with no PLE table),
# a single 128 GB DGX Spark has ample room for a large multi-sequence KV pool.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

# bf16 recurrent state for the 36 Gated DeltaNet linear-attention layers.
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-bfloat16}"

# LANGUAGE_MODEL_ONLY=1 (default) skips loading the 27-layer vision tower for
# text-only serving; set LANGUAGE_MODEL_ONLY=0 to serve multimodal image/video requests.
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"

# Optional MTP speculative decoding config, e.g.:
#   SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":1}'
#   SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":3}'
# Defaults to "none" (target-only decode, where CUDA graphs and cooperative MoE run at full speed).
SPEC_CONFIG="${SPEC_CONFIG:-none}"

REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
PROFILER_DIR="${PROFILER_DIR:-}"
DRY_RUN="${DRY_RUN:-0}"

if awk -v u="$GPU_MEM_UTIL" 'BEGIN{exit !(u > 0.90)}'; then
  echo "GPU_MEM_UTIL=$GPU_MEM_UTIL is above the recommended ceiling (0.90) for unified memory." >&2
  exit 2
fi

if [[ "$DRY_RUN" != "1" ]]; then
  if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    echo "missing $MODEL_DIR/config.json -- download the pack and run vllm-plugin/prepare_pack.sh" >&2
    exit 1
  fi
  if ! grep -q '"non_routed_exl3"' "$MODEL_DIR/config.json" 2>/dev/null; then
    echo "error: $MODEL_DIR/config.json does not look like a prepared vLLM pack (missing non_routed_exl3)." >&2
    echo "       run vllm-plugin/prepare_pack.sh first (creates ${MODEL_DIR%/}-vllm by default)." >&2
    exit 1
  fi
  if [[ ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
    echo "missing $MODEL_DIR/model.safetensors.index.json -- run vllm-plugin/prepare_pack.sh" >&2
    exit 1
  fi
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Enable fused cooperative MoE kernel (exl3_moe_coop) across all 48 uniform-(K_gu, K_d) layers
# and consecutive shard coalescing (in_proj_qkv [3072 -> 12288]) in Exl3LinearMethod.
export VLLM_EXL3_COOP="${VLLM_EXL3_COOP:-1}"
export VLLM_EXL3_COALESCE_SHARDS="${VLLM_EXL3_COALESCE_SHARDS:-1}"

ARGS=(
  serve "$MODEL_DIR"
  --served-model-name "$SERVED_NAME"
  --host "$HOST"
  --port "$PORT"
  --quantization exl3
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --enable-prefix-caching
  --reasoning-parser "$REASONING_PARSER"
  --enable-auto-tool-choice
  --tool-call-parser "$TOOL_CALL_PARSER"
)

if [[ "$LANGUAGE_MODEL_ONLY" == "1" || "$LANGUAGE_MODEL_ONLY" == "true" ]]; then
  ARGS+=(--language-model-only)
elif [[ "$LANGUAGE_MODEL_ONLY" == "0" || "$LANGUAGE_MODEL_ONLY" == "false" ]]; then
  # Cap multimodal dummy profile allocations so 27-layer vision tower profiling fits cleanly in 128 GB unified memory.
  LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\":1,\"video\":0}}"
  MM_MAX_BATCHED_TOKENS="${MM_MAX_BATCHED_TOKENS:-4096}"
  ARGS+=(--limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT" --max-num-batched-tokens "$MM_MAX_BATCHED_TOKENS")
else
  echo "LANGUAGE_MODEL_ONLY must be 1/0 or true/false, got '$LANGUAGE_MODEL_ONLY'" >&2
  exit 2
fi

if [[ -n "$MAMBA_SSM_DTYPE" && "$MAMBA_SSM_DTYPE" != "none" ]]; then
  ARGS+=(--mamba-ssm-cache-dtype "$MAMBA_SSM_DTYPE")
fi

if [[ -n "$SPEC_CONFIG" && "$SPEC_CONFIG" != "none" ]]; then
  ARGS+=(--speculative-config "$SPEC_CONFIG")
fi

if [[ -n "$PROFILER_DIR" ]]; then
  ARGS+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILER_DIR\"}")
fi

echo "VLLM_EXL3_COOP=$VLLM_EXL3_COOP VLLM_EXL3_COALESCE_SHARDS=$VLLM_EXL3_COALESCE_SHARDS LANGUAGE_MODEL_ONLY=$LANGUAGE_MODEL_ONLY GPU_MEM_UTIL=$GPU_MEM_UTIL MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS MAMBA_SSM_DTYPE=$MAMBA_SSM_DTYPE SPEC_CONFIG=${SPEC_CONFIG:-none}"
echo "vllm ${ARGS[*]}"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

exec vllm "${ARGS[@]}"
