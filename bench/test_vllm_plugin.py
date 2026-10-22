"""CPU unit tests for vllm-plugin scripts (patch_vllm_qwen3_5_vision_split.py, prepare_pack.sh, preflight.py)."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

VLLM_DIR = Path(__file__).resolve().parents[1] / "vllm-plugin"

PATCH_SPEC = importlib.util.spec_from_file_location(
    "patch_vllm_qwen3_5_vision_split", VLLM_DIR / "patch_vllm_qwen3_5_vision_split.py"
)
patch_vision = importlib.util.module_from_spec(PATCH_SPEC)
PATCH_SPEC.loader.exec_module(patch_vision)

PREFLIGHT_SPEC = importlib.util.spec_from_file_location("preflight", VLLM_DIR / "preflight.py")
preflight = importlib.util.module_from_spec(PREFLIGHT_SPEC)
PREFLIGHT_SPEC.loader.exec_module(preflight)


def write_safetensor(path: Path, tensors: dict):
    header, offset = {}, 0
    for name, (dtype, shape, size) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        handle.truncate(8 + len(raw) + 64)


class VllmPluginTests(unittest.TestCase):
    def test_vision_patch_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            qwen3_vl = Path(tmp) / "qwen3_vl.py"
            qwen3_vl.write_text(patch_vision.OLD)
            with patch.object(sys, "argv", ["patch_vllm_qwen3_5_vision_split.py", str(qwen3_vl)]):
                self.assertEqual(patch_vision.main(), 0)
            self.assertIn('".attn.q_proj.": None', qwen3_vl.read_text())
            self.assertTrue((Path(tmp) / "qwen3_vl.py.orig").is_file())
            with patch.object(sys, "argv", ["patch_vllm_qwen3_5_vision_split.py", str(qwen3_vl)]):
                self.assertEqual(patch_vision.main(), 0)

    def test_preflight_models_and_patches_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            vllm_root = Path(tmp) / "vllm"
            models = vllm_root / "model_executor/models"
            models.mkdir(parents=True)
            (models / "qwen3_5.py").write_text("class Qwen3_5MoeForConditionalGeneration:\n    pass\n")
            (models / "qwen3_5_mtp.py").write_text("class Qwen3_5MoeMTP:\n    pass\n")
            (models / "qwen3_vl.py").write_text(patch_vision.OLD)

            unpatched = preflight.check_vllm_models_and_patches(vllm_root)
            self.assertEqual(unpatched.status, "FAIL")
            self.assertIn("split vision q/k/v names not dropped", unpatched.detail)

            with patch.object(sys, "argv", ["patch_vllm_qwen3_5_vision_split.py", str(vllm_root)]):
                self.assertEqual(patch_vision.main(), 0)
            patched = preflight.check_vllm_models_and_patches(vllm_root)
            self.assertEqual(patched.status, "PASS", patched.detail)

    def test_prepare_pack_defaults_to_non_destructive_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = root / "Ornith-1.5-122B-EXL3-SAGE-2.58bpw"
            pack.mkdir()
            orig_config = {
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "mtp_num_hidden_layers": 1,
                "quantization_config": {
                    "quant_method": "exl3",
                    "bits": 2.58,
                    "head_bits": 8,
                    "codebook": "mul1",
                },
                "text_config": {
                    "num_hidden_layers": 48,
                    "full_attention_interval": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 256,
                },
            }
            (pack / "config.json").write_text(json.dumps(orig_config, indent=2) + "\n")
            orig_config_bytes = (pack / "config.json").read_bytes()

            index_obj = {
                "metadata": {"total_size": 2048},
                "weight_map": {
                    "model.layers.0.self_attn.q_proj.trellis": "model-00001-of-00006.safetensors",
                    "mtp.fc.trellis": "model-00006-of-00006.safetensors",
                },
            }
            (pack / "model.safetensors.index.json").write_text(json.dumps(index_obj, indent=2) + "\n")
            orig_index_bytes = (pack / "model.safetensors.index.json").read_bytes()
            for idx in range(1, 7):
                write_safetensor(
                    pack / f"model-{idx:05d}-of-00006.safetensors",
                    {"model.layers.0.self_attn.q_proj.trellis": ("I16", [16, 16, 64], 512)},
                )

            plugin_repo = root / "vllm-exl3"
            tools_dir = plugin_repo / "tools/exl3_pack_tools"
            tools_dir.mkdir(parents=True)
            (tools_dir / "qwen_pack_scan.py").write_text(
                "import json, os, sys\n"
                "target = sys.argv[1]\n"
                "scan = {\n"
                "  'pack_dir': target,\n"
                "  'suffixes_by_family': {'self_attn': ['trellis', 'suh', 'svh', 'mul1']},\n"
                "  'dense_k_by_prefix': {'lm_head': 8, 'mtp.fc': 4, 'model.layers.0.self_attn.q_proj': 4},\n"
                "  'uniform_moe_layers': list(range(48)),\n"
                "  'mixed_moe_layers': {},\n"
                "}\n"
                "with open(os.path.join(target, 'pack_scan.json'), 'w') as fh:\n"
                "  json.dump(scan, fh, indent=2)\n"
            )
            (tools_dir / "qwen_pack_config.py").write_text(
                "import json, os, sys\n"
                "target = sys.argv[1]\n"
                "scan = json.load(open(os.path.join(target, 'pack_scan.json')))\n"
                "cfg_path = os.path.join(target, 'config.json')\n"
                "cfg = json.load(open(cfg_path))\n"
                "q = cfg.get('quantization_config', {})\n"
                "cfg['quantization_config'] = {\n"
                "  'quant_method': 'exl3',\n"
                "  'derived_from_headers': True,\n"
                "  'codebook': q.get('codebook', 'mcg'),\n"
                "  'head_bits': q.get('head_bits'),\n"
                "  'uniform_moe_layers': scan['uniform_moe_layers'],\n"
                "  'mixed_moe_layers': scan['mixed_moe_layers'],\n"
                "  'non_routed_exl3': {\n"
                "    'layers': {'model.layers.0.self_attn.q_proj': {'bits': 4}},\n"
                "    'modules': ['model.layers.0.self_attn.q_proj'],\n"
                "  },\n"
                "}\n"
                "with open(cfg_path, 'w') as fh:\n"
                "  json.dump(cfg, fh, indent=2)\n"
            )
            (tools_dir / "regenerate_safetensors_index.py").write_text(
                "import os, shutil, sys\n"
                "target = sys.argv[1]\n"
                "idx = os.path.join(target, 'model.safetensors.index.json')\n"
                "nat = idx + '.native'\n"
                "if os.path.exists(nat):\n"
                "  shutil.copyfile(nat, idx)\n"
            )

            env = os.environ.copy()
            env.update(
                PACK_DIR=str(pack),
                PLUGIN_REPO=str(plugin_repo),
            )
            subprocess.run(["bash", str(VLLM_DIR / "prepare_pack.sh")], env=env, check=True)

            # Native pack must remain 100% untouched!
            self.assertEqual((pack / "config.json").read_bytes(), orig_config_bytes)
            self.assertEqual((pack / "model.safetensors.index.json").read_bytes(), orig_index_bytes)
            self.assertFalse((pack / "pack_scan.json").exists())

            # Non-destructive view must exist at ${PACK_DIR}-vllm
            view = Path(f"{pack}-vllm")
            self.assertTrue(view.is_dir())
            self.assertTrue((view / "model-00001-of-00006.safetensors").is_symlink())
            self.assertFalse((view / "config.json").is_symlink())
            view_cfg = json.loads((view / "config.json").read_text())
            qcfg = view_cfg["quantization_config"]
            self.assertEqual(qcfg["codebook"], "mul1")
            self.assertEqual(qcfg["head_bits"], 8)
            self.assertIn("mtp.fc", qcfg["non_routed_exl3"]["layers"])
            self.assertIn("mtp.fc", qcfg["non_routed_exl3"]["modules"])
            self.assertEqual(len(qcfg["uniform_moe_layers"]), 48)

            # Preflight pack check on the prepared view must pass
            res = preflight.check_pack(view)
            self.assertEqual(res.status, "PASS", res.detail)


if __name__ == "__main__":
    unittest.main()
