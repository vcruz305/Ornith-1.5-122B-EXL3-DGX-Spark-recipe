# Historical Benchmark Ledger — `vcruz305/Ornith-1.5-122B-EXL3-SAGE-2.58bpw` on DGX Spark (`GB10`)

All measurements below were recorded on a single NVIDIA DGX Spark (`GB10`, `sm_121`, 128 GB unified LPDDR5x, `121.7 GiB` visible to Linux) with `vllm==0.32.0`, `torch==2.13.0+cu130`, `vcruz305/vllm-exl3` (`0.5.0` @ `3275819`, compiled with `NVCC_APPEND_FLAGS="-DP2B_CB=2"` for the `mul1` codebook).

---

## 1. Kernel Microbenchmarks (`GB10`, `hidden=3072, moe_inter=1024, E=256, top_k=8, mul1`)

### A. Routed MoE Layer Execution (`exl3_moe_coop` vs `exl3_moe` vs `exl3_moe_mixedk`)

| Layer Family | `rows=1` (`exl3_moe` full) | `rows=1` (`exl3_moe_mixedk` `K2=2K`) | `rows=1` (`exl3_moe_coop` prealloc) | Speedup vs `exl3_moe` | Speedup vs `mixedk` | `rows=4` (`exl3_moe_coop`) | `rows=8` (`exl3_moe_coop`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Asymmetric `(kg=2, ku=2, kd=3)` (30/48 trunk layers)** | 447.92 µs | 401.70 µs | **135.29 µs** | **3.31×** | **2.97×** | **411.75 µs** | **756.17 µs** |
| **Symmetric `(kg=3, ku=3, kd=3)` (18/48 trunk layers)** | 272.42 µs | 352.34 µs | **167.69 µs** | **1.62×** | **2.10×** | **489.21 µs** | **855.85 µs** |
| **48-Layer Trunk Total per Step** | 18.34 ms | 18.39 ms | **7.08 ms** | **2.59×** | **2.60×** | **21.16 ms** | **38.09 ms** |

### B. Dense `LinearEXL3` Projections & Shard Coalescing (`rows=1`)

| Projection | Checkpoint Shape | `K` | Uncoalesced (`vLLM` 4-shard `in_proj_qkvz`) | Coalesced (`[3072->12288]` + `[3072->8192]`) | Savings / Layer |
|---|---|---:|---:|---:|---:|
| **GDN `in_proj_qkvz` (30 layers)** | `[3072 -> 12288] + [3072 -> 8192]` | 6 | 141.78 µs *(4 launches)* | **103.65 µs** *(2 launches)* | **-38.13 µs/layer** |
| **GDN `in_proj_qkvz` (6 layers)** | `[3072 -> 12288] + [3072 -> 8192]` | 8 | 284.42 µs *(4 launches)* | **225.10 µs** *(2 launches)* | **-59.32 µs/layer** |
| **36-Layer GDN Total** | — | 6 / 8 | 5.96 ms | **4.46 ms** | **-1.50 ms/step** |

---

## 2. End-to-End `vLLM 0.32.0 + vllm-exl3` A/B Ablation (`GB10`, `max_tokens=512`, `T=0.0`)

| Arm | `VLLM_EXL3_COOP` | `VLLM_EXL3_COALESCE_SHARDS` | `--mamba-ssm-cache-dtype` | Extra Knobs | Mean Decode (`tok/s`) | Peak Decode (`tok/s`) | Mean TTFT (`ms`) | CUDA Graph Capture |
|---|---:|---:|---|---|---:|---:|---:|---|
| **1. Baseline** | `0` | `0` | `auto` (`float32`) | — | 22.84 | 22.90 | 385.0 | 11.0 s (`exl3_moe` cooperative-launch instability) |
| **2. `+ Coop MoE`** | `1` | `0` | `auto` (`float32`) | `_COOP_TEMP_CACHE` | 29.40 | 29.47 | 111.5 | **2.0 s (`+28.7%` decode)** |
| **3. `+ Coalesce Shards`** | `1` | `1` | `auto` (`float32`) | — | 30.48 | 30.54 | 110.2 | **2.0 s (`+33.4%` decode)** |
| **4. `+ BF16 SSM (`serve.sh` default)** | `1` | `1` | `bfloat16` | — | **30.51** | **30.57** | **109.1** | **2.0 s (`+33.6%` decode, `-50%` SSM state RAM)** |
| **5. `+ MTP k=1` (Code / Mean)** | `1` | `1` | `bfloat16` | `num_speculative_tokens=1` | **32.60** *(code)* / **30.38** *(mean)* | **32.99** | 124.2 | 4.0 s (`88.6%` code acceptance) |
| **6. `EXL3_INT8_GEMV_MAX_K=8` test** | `1` | `1` | `bfloat16` | `EXL3_INT8_GEMV_MAX_K=8` | 29.74 | 29.83 | 111.2 | `lm_head` (`K=8`) is `82.5 µs` faster on `exl3_gemm_coop` (`MAX_K=6` wins) |
| **7. `RECONSTRUCT_MIN_ROWS=145` test** | `1` | `1` | `bfloat16` | `VLLM_EXL3_RECONSTRUCT_MIN_ROWS=145` | 30.50 | 30.55 | 109.4 | Default `16` is `+9.2%` faster on 2K prefill (`1,119.1` vs `1,024.9 tok/s`) |
