# Ornith-1.5-122B EXL3 on NVIDIA DGX Spark — Documentation

Documentation and interactive overview for serving
[`vcruz305/Ornith-1.5-122B-EXL3-SAGE-2.58bpw`](https://huggingface.co/vcruz305/Ornith-1.5-122B-EXL3-SAGE-2.58bpw)
(`Qwen3_5MoeForConditionalGeneration`, base model [`ornith-ai/Ornith-1.5-122B`](https://huggingface.co/ornith-ai/Ornith-1.5-122B))
on one NVIDIA DGX Spark (`GB10`, 128 GB unified memory) via `vllm==0.32.0` + `vcruz305/vllm-exl3`.

## Guides

- [`tool-calling.md`](tool-calling.md) — Qwen XML (`qwen3_xml`) tool-calling semantics, argument types, streaming, and SDK validation.
- [`service.md`](service.md) — User-systemd deployment (`ornith-vllm.service`) for `vllm-plugin/serve.sh`.
- [`memory.md`](memory.md) — Header-based weight and KV cache estimation (`vllm-plugin/model_memory.py`).
- [`api-validation.md`](api-validation.md) — Serial API resilience (`bench/api_resilience.py`) and automatic tool selection (`bench/auto_compatibility.py`) checks.

## Render PNGs locally

```sh
python -m pip install playwright
python -m playwright install chromium
python docs/render_benchmark.py
```

This renders the 1920x1080 overview cards to `benchmark-renders/` without requiring a GPU or model weights.
