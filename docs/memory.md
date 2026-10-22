# Memory advisory and KV cache sizing

The recipe's [memory helper](../vllm-plugin/model_memory.py) reads
safetensors headers and the model configuration without loading weights or
initializing CUDA. It estimates pack storage, KV cache pool backing,
and system memory headroom.

## What is counted

Because `vcruz305/Ornith-1.5-122B-EXL3-SAGE-2.58bpw` (`Qwen3_5MoeForConditionalGeneration`)
has no n-gram PLE table (`ngram_bytes == 0`), the full `42.01 GiB` checkpoint (`41.00 GiB`
language model + `0.59 GiB` MTP + `0.37 GiB` vision tower across 6 shards) leaves over `50 GiB`
for the KV cache pool at `GPU_MEM_UTIL=0.80` on a 128 GB DGX Spark.

| Component | Storage |
| --- | --- |
| `other_weight_bytes` | Language model (`41.00 GiB`), MTP (`0.59 GiB`), and vision tower (`0.37 GiB`) tensors (`42.01 GiB` total across 6 shards). |
| BF16 KV cache (`vLLM`) | `12` full-attention layers (`2` KV heads x `256` `head_dim`) = `24 KiB/token` (`~6.0 GiB` per `262,144` tokens). |
| K8/V8 KV cache (`model_memory.py` estimate) | `12` full-attention layers + `1` MTP layer (`~3.45 GiB` at `262,144` tokens, `~13.81 GiB` at `1,048,576` tokens). |

## Usage

Run the advisory against the downloaded or prepared pack directory:

```bash
python3 vllm-plugin/model_memory.py \
  --model ~/models/Ornith-1.5-122B-EXL3-SAGE-2.58bpw-vllm \
  --cache-size 262144 --draft-mode mtp --ngram-ram false \
  --slack-gib 10 --json > memory-advisory.json
```

CPU checks for the geometry and controller-facing fields:

```bash
python3 -m unittest discover -s bench -p test_model_memory.py -v
```
