# API resilience and automatic tool selection checks

The recipe includes two serial, standard-library clients for checking a running OpenAI-compatible `/v1` server:

- [`bench/api_resilience.py`](../bench/api_resilience.py): raw completions, model identity, forced tools with reasoning, nullable strings, invalid-request recovery, and an early streaming disconnect.
- [`bench/auto_compatibility.py`](../bench/auto_compatibility.py): ordinary responses with tools available, reasoning, and explicit client output constraints.

Run these **between performance measurements**, while the server is otherwise idle. Their wall times describe individual functional checks. Use the [benchmark clients](../bench/README.md) for throughput and concurrency measurements.

Both clients use Python 3.10 or later with no additional packages. Every prompt, function declaration, and requested argument is a synthetic fixture. The clients never execute a returned function or contact an external tool.

## Run the checks

Start the server (`bash vllm-plugin/serve.sh`) and run:

```bash
python3 bench/api_resilience.py \
  --base-url http://127.0.0.1:8899/v1 \
  --model Ornith-1.5-122B-EXL3 \
  --label candidate \
  --output results/candidate-api-resilience.json

python3 bench/auto_compatibility.py \
  --base-url http://127.0.0.1:8899/v1 \
  --model Ornith-1.5-122B-EXL3 \
  --label candidate \
  --output results/candidate-auto-compatibility.json
```

**Choose a new output filename for each run.** An existing file, directory, or dangling symlink is refused before the first HTTP request.

## Offline tests

From the recipe root:

```bash
PYTHONPATH=bench python3 -m unittest -v test_api_resilience test_auto_compatibility
```

The recipe-wide CPU discovery command also includes these tests:

```bash
python3 -m unittest discover -s bench -p 'test_*.py' -v
```
