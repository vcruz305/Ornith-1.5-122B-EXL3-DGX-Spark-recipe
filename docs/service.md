# Run the recipe as a systemd user service

The user service starts `vllm-plugin/serve.sh` under the account that owns the model and `vLLM` environment. It binds to `127.0.0.1:8899` by default.

## Configure and install

Create `~/.config/systemd/user/ornith-vllm.service`:

```ini
[Unit]
Description=Ornith-1.5-122B EXL3 vLLM Server (127.0.0.1:8899)
After=network-online.target

[Service]
Type=exec
Environment=PATH=%h/venvs/vllm-exl3/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
Environment=MODEL_DIR=%h/models/Ornith-1.5-122B-EXL3-SAGE-2.58bpw-vllm
Environment=HOST=127.0.0.1
Environment=PORT=8899
Environment=GPU_MEM_UTIL=0.80
Environment=MAX_MODEL_LEN=262144
Environment=MAX_NUM_SEQS=8
Environment=VLLM_EXL3_COOP=1
Environment=VLLM_EXL3_COALESCE_SHARDS=1
WorkingDirectory=%h/Ornith-1.5-122B-EXL3-DGX-Spark-recipe
ExecStart=/usr/bin/bash %h/Ornith-1.5-122B-EXL3-DGX-Spark-recipe/vllm-plugin/serve.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

Run `preflight.py` first:

```bash
~/venvs/vllm-exl3/bin/python vllm-plugin/preflight.py \
  --pack ~/models/Ornith-1.5-122B-EXL3-SAGE-2.58bpw-vllm
systemd-analyze --user verify "$HOME/.config/systemd/user/ornith-vllm.service"
```

Then enable and start the unit:

```bash
systemctl --user daemon-reload
systemctl --user enable --now ornith-vllm.service
systemctl --user status ornith-vllm.service
journalctl --user -u ornith-vllm.service -n 100 --no-pager
```

## Readiness and validation

Wait for the expected model ID (`Ornith-1.5-122B-EXL3`) in the API:

```bash
curl --fail --silent --show-error http://127.0.0.1:8899/v1/models
```

Then run the tool checks and a short strict-client benchmark:

```bash
python3 bench/tool_smoke.py --model Ornith-1.5-122B-EXL3 \
  --case auto,no_args,strings,typed,required,named,none --mode both \
  --output results/service-tools.json
python3 bench/bench_v1.py --model Ornith-1.5-122B-EXL3 \
  --suite code --repeat 1 --warmup 0 --max-tokens 128 \
  --output results/service-bench.json
```
