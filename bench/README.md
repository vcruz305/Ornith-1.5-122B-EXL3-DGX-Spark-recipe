# API performance and tool calling validation

These clients exercise the deployed `vLLM` `/v1` endpoint. The performance
clients and `tool_smoke.py` use the Python standard library and require no access to
a GPU themselves. The optional `sdk_smoke.py` uses the official OpenAI Python SDK in
a separate client environment.

See [Tool calling on the Spark recipe](../docs/tool-calling.md) for supported
request modes, argument and streaming contracts, error handling, and commands
for the five-case SDK round-trip check.

## Before measuring

Keep the baseline and candidate on the same model pack and workload. Use a quiet
machine with one model server. Complete a benchmark before changing the server
or its environment. Do not run correctness tests at the same time as a throughput
sample.

## Single stream

For each of code, DevOps, and prose, run one warmup followed by three measured
requests:

```bash
python3 bench/bench_v1.py \
  --model Ornith-1.5-122B-EXL3 \
  --suite all --warmup 1 --repeat 3 --max-tokens 400 \
  --label ornith-2.58bpw --run-id comparison-001 \
  --output results/ornith-2.58bpw.json
```

By default, each measured request changes an early prompt prefix. The result
records actual `cached_prompt_tokens` so caching remains visible. To measure
reused prompts instead:

```bash
python3 bench/bench_v1.py --suite code --cache-mode warm \
  --context-tokens 24000 --repeat 3 --output results/warm-24k.json
```

### Metrics

| Field | Meaning |
|---|---|
| `client_end_to_end_tok_s` | Actual completion tokens divided by the whole client request time, including queueing, prefill, and HTTP overhead. |
| `ttft_s` | Time from starting the HTTP request to the first content, reasoning, or tool delta. Role chunks and SSE keepalive comments do not count. |
| `client_decode_tok_s_estimate` | Decode throughput estimate using first/last output events and completion count. |
| `cached_prompt_tokens` | The server-reported cache reuse count, or null if unavailable. |

## Concurrent serving

The following runs one, four, and eight simultaneous requests:

```bash
python3 bench/concurrency.py candidate 1,4,8 3000 400 3 \
  --warmup 1 --model Ornith-1.5-122B-EXL3 \
  --output results/concurrency.json
```

## Long-context retrieval

Check exact retrieval across early, middle, and late positions in a cold prompt:

```bash
python3 bench/long_context.py \
  --model Ornith-1.5-122B-EXL3 \
  --tokenizer ~/models/Ornith-1.5-122B-EXL3-SAGE-2.58bpw-vllm/tokenizer.json \
  --context-tokens 240000 \
  --output results/long-240k.json
```

## Tool calling

```bash
python3 bench/tool_smoke.py --model Ornith-1.5-122B-EXL3 \
  --mode both --output results/tools.json
```

All tools are synthetic fixtures; no shell commands or external requests are executed.

## CPU regression checks

```bash
python3 -m unittest discover -s bench -p 'test_*.py' -v
bash -n vllm-plugin/setup_env.sh vllm-plugin/prepare_pack.sh vllm-plugin/serve.sh
```
