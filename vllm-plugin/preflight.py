#!/usr/bin/env python3
"""
Environment preflight checker for vcruz305/Ornith-1.5-122B-EXL3-SAGE-2.58bpw
serving on vLLM 0.32.0 + vcruz305/vllm-exl3.

This script validates the environment in under 5 seconds without allocating GPU
memory or loading the 42 GiB checkpoint.

Checks verify:
  1. Python environment (>= 3.10) and platform.
  2. PyTorch installation, CUDA version, and Blackwell/GB10 arch support (sm_120/sm_121).
  3. vLLM version (0.32.0).
  4. exllamav3_ext compiled kernels (exl3_moe_coop, exl3_moe, exl3_mgemm, exl3_gemm, had_r_128, hgemm, reconstruct).
  5. vllm_exl3 quantization registration and compiled vllm_exl3_c extension.
  6. vLLM model classes (Qwen3_5MoeForConditionalGeneration + Qwen3_5MoeMTP) and Qwen3.5 vision split patch.
  7. Prepared pack integrity (when --pack is passed): codebook=mul1, head_bits=8, mtp.fc, non_routed_exl3, and safetensors index.
  8. Available system memory (/proc/meminfo on Linux).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import NamedTuple, Optional


class Check(NamedTuple):
    """Result of a single environment check."""
    status: str  # PASS, FAIL, WARN, SKIP
    name: str
    detail: str


def get_vllm_dir(vllm_dir_arg: Optional[str]) -> Optional[Path]:
    """Find vLLM installation directory."""
    if vllm_dir_arg:
        return Path(vllm_dir_arg)
    try:
        spec = importlib.util.find_spec("vllm")
        if spec and spec.origin:
            return Path(spec.origin).parent
    except (ImportError, AttributeError, ValueError):
        pass
    return None


def check_platform() -> Check:
    """Check Python version, machine architecture, and OS."""
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    machine = platform.machine()
    system = platform.system()
    detail = f"Python {py_version}, {machine}, {system}"

    if sys.version_info < (3, 10):
        return Check("FAIL", "platform", f"{detail} (Python 3.10 or later required)")

    return Check("PASS", "platform", detail)


def check_torch() -> Check:
    """Check PyTorch installation, version, and CUDA configuration (sm_120/sm_121)."""
    try:
        import torch
    except ImportError:
        return Check(
            "FAIL",
            "torch",
            "not importable\nremedy: activate the serving venv, or run bash vllm-plugin/setup_env.sh",
        )

    try:
        torch_version = torch.__version__
        cuda_version = torch.version.cuda or "None"
        arch_list = torch.cuda.get_arch_list()
    except Exception as e:
        return Check("FAIL", "torch", f"error accessing properties: {e}")

    detail = f"version {torch_version}, CUDA {cuda_version}, arches {arch_list}"

    has_sm120 = any("sm_120" in str(a) for a in arch_list)
    has_sm121 = any("sm_121" in str(a) for a in arch_list)

    if not (has_sm120 or has_sm121):
        return Check(
            "WARN",
            "torch",
            f"{detail} (sm_120 and sm_121 not found; GB10 is sm_121 and runs sm_120/sm_121 binaries)",
        )

    return Check("PASS", "torch", detail)


def check_vllm() -> Check:
    """Check vLLM availability and version (0.32.0)."""
    try:
        import vllm
    except ImportError:
        return Check(
            "FAIL",
            "vllm",
            "not importable\nremedy: pip install vllm==0.32.0 (or run bash vllm-plugin/setup_env.sh)",
        )

    version = vllm.__version__
    detail = f"version {version}"

    if version == "0.32.0" or version.startswith("0.32.0"):
        return Check("PASS", "vllm", detail)

    return Check("WARN", "vllm", f"{detail} (expected known-good version: 0.32.0)")


def check_exllamav3_ext() -> Check:
    """Check exllamav3_ext kernels (exl3_moe_coop, exl3_moe, exl3_mgemm) are available."""
    try:
        import torch  # noqa: F401 - must precede exllamav3_ext for libtorch RPATH
        import exllamav3_ext
    except ImportError:
        return Check(
            "FAIL",
            "exllamav3_ext",
            "not importable\nremedy: build vcruz305/exllamav3 from source for sm_121 (bash vllm-plugin/setup_env.sh)",
        )

    required_kernels = [
        "exl3_moe_coop",
        "exl3_moe",
        "exl3_mgemm",
        "exl3_gemm",
        "had_r_128",
        "hgemm",
        "reconstruct",
    ]

    missing = [k for k in required_kernels if not hasattr(exllamav3_ext, k)]
    if missing:
        remedy = (
            "build vcruz305/exllamav3 (master) from source for sm_121; "
            "exl3_moe_coop, exl3_moe, and exl3_mgemm are required for Ornith-1.5-122B EXL3"
        )
        return Check("FAIL", "exllamav3_ext", f"missing kernels: {missing}\nremedy: {remedy}")

    return Check("PASS", "exllamav3_ext", "exl3_moe_coop, exl3_moe, exl3_mgemm, and dense kernels present")


def check_vllm_exl3() -> Check:
    """Check vllm_exl3 import, compiled vllm_exl3_c extension, and quantization registry."""
    try:
        import torch  # noqa: F401
        import vllm_exl3  # noqa: F401
    except ImportError:
        return Check(
            "FAIL",
            "vllm_exl3",
            'not importable\nremedy: NVCC_APPEND_FLAGS="-DP2B_CB=2" pip install --no-build-isolation git+https://github.com/vcruz305/vllm-exl3@main',
        )

    c_ext_ok = False
    for mod_name in ("vllm_exl3.vllm_exl3_c", "vllm_exl3_c", "vllm_exl3._C"):
        try:
            if importlib.util.find_spec(mod_name) is not None:
                c_ext_ok = True
                break
        except (ImportError, ValueError):
            continue
    if not c_ext_ok:
        try:
            from vllm_exl3 import exl3 as _exl3_mod
            c_ext_ok = bool(getattr(_exl3_mod, "_C_EXT", None) or getattr(_exl3_mod, "vllm_exl3_c", None))
        except Exception:
            pass
    if not c_ext_ok:
        pkg_dir = Path(vllm_exl3.__file__).resolve().parent
        so_files = list(pkg_dir.glob("*exl3_c*.so")) + list(pkg_dir.parent.glob("*exl3_c*.so"))
        c_ext_ok = bool(so_files)

    if not c_ext_ok:
        return Check(
            "FAIL",
            "vllm_exl3",
            'vllm_exl3_c compiled extension not found\nremedy: NVCC_APPEND_FLAGS="-DP2B_CB=2" pip install --no-build-isolation git+https://github.com/vcruz305/vllm-exl3@main',
        )

    try:
        from vllm.model_executor.layers.quantization import get_quantization_config
        get_quantization_config("exl3")
    except Exception as e:
        return Check("FAIL", "vllm_exl3", f"exl3 not in quantization registry: {e}")

    return Check("PASS", "vllm_exl3", "importable, vllm_exl3_c present, and registered")


def check_vllm_models_and_patches(vllm_dir: Path) -> Check:
    """Check Qwen3_5MoeForConditionalGeneration, Qwen3_5MoeMTP, and vision split patch."""
    models_dir = vllm_dir / "model_executor" / "models"
    qwen35_file = models_dir / "qwen3_5.py"
    qwen35_mtp_file = models_dir / "qwen3_5_mtp.py"
    qwen3_vl_file = models_dir / "qwen3_vl.py"

    issues = []
    if not qwen35_file.is_file():
        issues.append(("Qwen3_5MoeForConditionalGeneration (qwen3_5.py) missing", "install vllm==0.32.0"))
    else:
        src = qwen35_file.read_text(encoding="utf-8", errors="replace")
        if "class Qwen3_5MoeForConditionalGeneration" not in src:
            issues.append(("Qwen3_5MoeForConditionalGeneration not found in qwen3_5.py", "install vllm==0.32.0"))

    if not qwen35_mtp_file.is_file():
        issues.append(("Qwen3_5MoeMTP (qwen3_5_mtp.py) missing", "install vllm==0.32.0"))
    else:
        mtp_src = qwen35_mtp_file.read_text(encoding="utf-8", errors="replace")
        if "class Qwen3_5MoeMTP" not in mtp_src:
            issues.append(("Qwen3_5MoeMTP not found in qwen3_5_mtp.py", "install vllm==0.32.0"))

    if not qwen3_vl_file.is_file():
        issues.append(("qwen3_vl.py missing", "install vllm==0.32.0"))
    else:
        vl_src = qwen3_vl_file.read_text(encoding="utf-8", errors="replace")
        if ".attn.k_proj." not in vl_src:
            issues.append(
                (
                    "split vision q/k/v names not dropped in qwen3_vl.py",
                    "python3 vllm-plugin/patch_vllm_qwen3_5_vision_split.py <site-packages/vllm>",
                )
            )

    if issues:
        detail_parts = []
        for desc, remedy in issues:
            detail_parts.append(f"missing: {desc}")
            detail_parts.append(f"remedy: {remedy}")
        return Check("FAIL", "vllm models", "\n".join(detail_parts))

    return Check(
        "PASS",
        "vllm models",
        "Qwen3_5MoeForConditionalGeneration + Qwen3_5MoeMTP + vision split patch present",
    )


def check_pack(pack_path: Path) -> Check:
    """Check prepared Ornith-1.5-122B EXL3 pack integrity."""
    if not pack_path.exists():
        return Check(
            "FAIL",
            "pack",
            f"path does not exist: {pack_path}\nremedy: bash vllm-plugin/prepare_pack.sh",
        )

    config_json = pack_path / "config.json"
    if not config_json.exists():
        return Check("FAIL", "pack", "config.json not found\nremedy: bash vllm-plugin/prepare_pack.sh")

    try:
        config = json.loads(config_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return Check("FAIL", "pack", f"config.json invalid: {e}\nremedy: bash vllm-plugin/prepare_pack.sh")

    arches = config.get("architectures") or []
    if "Qwen3_5MoeForConditionalGeneration" not in arches:
        return Check(
            "FAIL",
            "pack",
            f"architectures={arches}, expected ['Qwen3_5MoeForConditionalGeneration']",
        )

    quant_config = config.get("quantization_config") or {}
    if not quant_config and isinstance(config.get("text_config"), dict):
        quant_config = config["text_config"].get("quantization_config") or {}

    quant_method = quant_config.get("quant_method")
    if quant_method != "exl3":
        return Check(
            "FAIL",
            "pack",
            f"quantization_config.quant_method is '{quant_method}', expected 'exl3'\nremedy: bash vllm-plugin/prepare_pack.sh",
        )

    codebook = quant_config.get("codebook")
    if codebook != "mul1":
        return Check(
            "FAIL",
            "pack",
            f"quantization_config.codebook is '{codebook}', expected 'mul1'\nremedy: bash vllm-plugin/prepare_pack.sh",
        )

    head_bits = quant_config.get("head_bits")
    if head_bits != 8:
        return Check(
            "FAIL",
            "pack",
            f"quantization_config.head_bits is '{head_bits}', expected 8\nremedy: bash vllm-plugin/prepare_pack.sh",
        )

    non_routed = quant_config.get("non_routed_exl3")
    if not isinstance(non_routed, dict) or not isinstance(non_routed.get("layers"), dict):
        return Check(
            "FAIL",
            "pack",
            "quantization_config.non_routed_exl3 missing\nremedy: bash vllm-plugin/prepare_pack.sh",
        )

    nr_layers = non_routed.get("layers") or {}
    nr_modules = non_routed.get("modules") or []
    if "mtp.fc" not in nr_layers and "mtp.fc" not in nr_modules:
        return Check(
            "FAIL",
            "pack",
            "mtp.fc missing from quantization_config.non_routed_exl3\nremedy: bash vllm-plugin/prepare_pack.sh",
        )

    safetensors_index = pack_path / "model.safetensors.index.json"
    if not safetensors_index.exists():
        return Check(
            "FAIL",
            "pack",
            "model.safetensors.index.json not found\nremedy: bash vllm-plugin/prepare_pack.sh",
        )

    return Check(
        "PASS",
        "pack",
        f"exl3 prepared (codebook={codebook}, head_bits={head_bits}, mtp.fc mapped, {len(nr_layers)} non_routed entries)",
    )


def check_memory() -> Check:
    """Check available system memory (Linux only)."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    mem_kb = int(line.split()[1])
                    mem_gb = mem_kb / (1024 ** 2)
                    detail = f"{mem_gb:.1f} GiB available"

                    if mem_gb < 60:
                        return Check(
                            "WARN",
                            "memory",
                            f"{detail} (Ornith-1.5-122B EXL3 2.58bpw weights need ~41 GiB plus KV cache; drop caches or stop competing workloads)",
                        )

                    return Check("PASS", "memory", detail)
    except (FileNotFoundError, OSError):
        return Check("SKIP", "memory", "not available on this platform")

    return Check("SKIP", "memory", "MemAvailable not found in /proc/meminfo")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Environment preflight checker for Ornith-1.5-122B EXL3 serving on vLLM."
    )
    parser.add_argument(
        "--pack",
        type=str,
        default=None,
        help="Path to prepared model pack; skips pack checks when absent.",
    )
    parser.add_argument(
        "--vllm-dir",
        type=str,
        default=None,
        help="Path to vLLM installation directory; auto-detected when absent.",
    )

    args = parser.parse_args()

    checks = [
        check_platform(),
        check_torch(),
        check_vllm(),
        check_exllamav3_ext(),
        check_vllm_exl3(),
    ]

    vllm_dir = get_vllm_dir(args.vllm_dir)
    if vllm_dir:
        checks.append(check_vllm_models_and_patches(vllm_dir))
    else:
        checks.append(Check("FAIL", "vllm models", "cannot locate vLLM directory"))

    if args.pack:
        checks.append(check_pack(Path(args.pack)))
    else:
        checks.append(Check("SKIP", "pack", "not specified"))

    checks.append(check_memory())

    max_status_len = max(len(c.status) for c in checks)
    max_name_len = max(len(c.name) for c in checks)

    for check in checks:
        status_part = check.status.ljust(max_status_len)
        name_part = check.name.ljust(max_name_len)
        lines = check.detail.split("\n")
        print(f"{status_part}  {name_part}  {lines[0]}")
        indent = " " * (max_status_len + max_name_len + 4)
        for extra in lines[1:]:
            print(f"{indent}{extra}")

    pass_count = sum(1 for c in checks if c.status == "PASS")
    fail_count = sum(1 for c in checks if c.status == "FAIL")
    warn_count = sum(1 for c in checks if c.status == "WARN")
    skip_count = sum(1 for c in checks if c.status == "SKIP")

    print()
    print(f"SUMMARY: {pass_count} passed, {fail_count} failed, {warn_count} warned, {skip_count} skipped")

    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"FAIL  uncaught exception  {type(e).__name__}: {e}")
        sys.exit(1)
