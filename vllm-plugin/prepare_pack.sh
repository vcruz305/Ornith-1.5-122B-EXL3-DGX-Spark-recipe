#!/usr/bin/env bash
# Prepare vcruz305/Ornith-1.5-122B-EXL3-SAGE-2.58bpw for vllm-exl3's native
# Qwen3_5MoeForConditionalGeneration + Qwen3_5MoeMTP loader.
#
# By default (IN_PLACE=0), this script creates a non-destructive symlinked view at
# PACK_VIEW_DIR (default: "${PACK_DIR}-vllm") so the downloaded pack at PACK_DIR is
# NEVER modified in place!
#
# Set IN_PLACE=1 (or pass --in-place) only if you want to rewrite PACK_DIR in place
# (backups: config.json.native and model.safetensors.index.json.native).
#
# Runs in order:
#   1. qwen_pack_scan.py             -> writes pack_scan.json
#   2. Enrich codebook/head_bits + qwen_pack_config.py -> writes config.json (quantization_config)
#   3. regenerate_safetensors_index.py -> writes model.safetensors.index.json
set -euo pipefail

IN_PLACE="${IN_PLACE:-0}"
PACK_DIR="${PACK_DIR:-}"
PACK_VIEW_DIR="${PACK_VIEW_DIR:-}"
PLUGIN_REPO="${PLUGIN_REPO:-}"

show_help() {
  cat <<EOF
usage: PACK_DIR=/path/to/pack PLUGIN_REPO=/path/to/vllm-exl3 bash vllm-plugin/prepare_pack.sh [OPTIONS]

Prepare an EXL3 pack for vllm-exl3. Defaults to creating a non-destructive symlinked view
at \${PACK_DIR}-vllm so the downloaded pack directory remains untouched.

OPTIONS:
  --pack-dir PATH       Source native EXL3 pack directory (or set PACK_DIR)
  --view-dir PATH       Destination symlinked view directory (default: <PACK_DIR>-vllm; or set PACK_VIEW_DIR)
  --in-place            Rewrite PACK_DIR in place instead of creating a symlinked view (or set IN_PLACE=1)
  --plugin-repo PATH    Path to vcruz305/vllm-exl3 checkout (or set PLUGIN_REPO)
  -h, --help            Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pack-dir)
      PACK_DIR="$2"
      shift 2
      ;;
    --view-dir)
      PACK_VIEW_DIR="$2"
      IN_PLACE=0
      shift 2
      ;;
    --in-place)
      IN_PLACE=1
      shift
      ;;
    --plugin-repo)
      PLUGIN_REPO="$2"
      shift 2
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "error: unknown option '$1'" >&2
      show_help >&2
      exit 1
      ;;
  esac
done

: "${PACK_DIR:?set PACK_DIR (or --pack-dir) to the downloaded vcruz305/Ornith-1.5-122B-EXL3-SAGE-2.58bpw directory}"
: "${PLUGIN_REPO:?set PLUGIN_REPO (or --plugin-repo) to a checkout of vcruz305/vllm-exl3}"

TOOLS="$PLUGIN_REPO/tools/exl3_pack_tools"
for f in qwen_pack_scan.py qwen_pack_config.py regenerate_safetensors_index.py; do
  if [[ ! -f "$TOOLS/$f" ]]; then
    echo "missing $TOOLS/$f -- is PLUGIN_REPO a checkout of vcruz305/vllm-exl3?" >&2
    exit 1
  fi
done

if [[ ! -f "$PACK_DIR/config.json" ]]; then
  echo "missing $PACK_DIR/config.json -- is PACK_DIR the pack directory?" >&2
  exit 1
fi

PACK_DIR="$(cd "$PACK_DIR" && pwd)"

if [[ "$IN_PLACE" == "1" ]]; then
  TARGET_DIR="$PACK_DIR"
  echo "mode: in-place preparation in $TARGET_DIR (backups: *.native)"
else
  PACK_VIEW_DIR="${PACK_VIEW_DIR:-${PACK_DIR%/}-vllm}"
  mkdir -p "$PACK_VIEW_DIR"
  PACK_VIEW_DIR="$(cd "$PACK_VIEW_DIR" && pwd)"
  if [[ "$PACK_VIEW_DIR" == "$PACK_DIR" ]]; then
    echo "PACK_VIEW_DIR ($PACK_VIEW_DIR) equals PACK_DIR; pass IN_PLACE=1 for in-place preparation or choose a distinct PACK_VIEW_DIR." >&2
    exit 1
  fi
  echo "mode: non-destructive symlink view at $PACK_VIEW_DIR (source: $PACK_DIR)"
  # Symlink all pack artifacts except config.json and model.safetensors.index.json,
  # which are copied so qwen_pack_config.py and regenerate_safetensors_index.py write
  # into PACK_VIEW_DIR without mutating PACK_DIR.
  find "$PACK_VIEW_DIR" -mindepth 1 -maxdepth 1 -type l -delete
  for f in "$PACK_DIR"/* "$PACK_DIR"/.*; do
    [[ -e "$f" ]] || continue
    b="$(basename "$f")"
    case "$b" in
      .|..|.cache|.git|config.json|config.json.native|model.safetensors.index.json|model.safetensors.index.json.native|pack_scan.json)
        continue
        ;;
    esac
    ln -sfn "$f" "$PACK_VIEW_DIR/$b"
  done
  if [[ -f "$PACK_DIR/config.json.native" ]]; then
    cp -f "$PACK_DIR/config.json.native" "$PACK_VIEW_DIR/config.json"
    cp -f "$PACK_DIR/config.json.native" "$PACK_VIEW_DIR/config.json.native"
  else
    cp -f "$PACK_DIR/config.json" "$PACK_VIEW_DIR/config.json"
    cp -f "$PACK_DIR/config.json" "$PACK_VIEW_DIR/config.json.native"
  fi
  if [[ -f "$PACK_DIR/model.safetensors.index.json.native" ]]; then
    cp -f "$PACK_DIR/model.safetensors.index.json.native" "$PACK_VIEW_DIR/model.safetensors.index.json"
    cp -f "$PACK_DIR/model.safetensors.index.json.native" "$PACK_VIEW_DIR/model.safetensors.index.json.native"
  elif [[ -f "$PACK_DIR/model.safetensors.index.json" ]]; then
    cp -f "$PACK_DIR/model.safetensors.index.json" "$PACK_VIEW_DIR/model.safetensors.index.json"
    cp -f "$PACK_DIR/model.safetensors.index.json" "$PACK_VIEW_DIR/model.safetensors.index.json.native"
  fi
  TARGET_DIR="$PACK_VIEW_DIR"
fi

echo "1/3 scanning pack headers in $TARGET_DIR ..."
python3 "$TOOLS/qwen_pack_scan.py" "$TARGET_DIR"

echo "2/3 rewriting $TARGET_DIR/config.json quantization_config ..."
# Ensure config.json.native backup exists first, then populate codebook (mul1) and
# head_bits (lm_head K=8) from pack_scan.json if the minimal native quantization_config
# omitted them, so qwen_pack_config.py emits the exact mul1 / head_bits=8 spec.
python3 - "$TARGET_DIR" <<'PY'
import json
import os
import shutil
import sys

target = sys.argv[1]
cfg_path = os.path.join(target, "config.json")
backup_path = cfg_path + ".native"
if not os.path.exists(backup_path):
    shutil.copyfile(cfg_path, backup_path)

scan = json.load(open(os.path.join(target, "pack_scan.json"), encoding="utf-8"))
cfg = json.load(open(cfg_path, encoding="utf-8"))

suffixes = set()
for fam_suffixes in (scan.get("suffixes_by_family") or {}).values():
    suffixes.update(fam_suffixes)

detected_codebook = "mul1" if ("mul1" in suffixes and "mcg" not in suffixes) else "mcg"
dense_k = scan.get("dense_k_by_prefix") or {}
detected_head_bits = int(dense_k.get("lm_head", 8))

for container in (cfg, cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else None):
    if not isinstance(container, dict):
        continue
    q = container.get("quantization_config")
    if isinstance(q, dict):
        target_q = q.get("native_quantization_config") if isinstance(q.get("native_quantization_config"), dict) else q
        target_q.setdefault("codebook", detected_codebook)
        target_q.setdefault("head_bits", detected_head_bits)

with open(cfg_path, "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, indent=2)
    fh.write("\n")
PY

python3 "$TOOLS/qwen_pack_config.py" "$TARGET_DIR"

# Ensure mtp.fc and model.fc are both mapped in non_routed_exl3 when mtp.fc is quantized in pack_scan.json
python3 - "$TARGET_DIR" <<'PY'
import json
import os
import sys

target = sys.argv[1]
cfg_path = os.path.join(target, "config.json")
scan = json.load(open(os.path.join(target, "pack_scan.json"), encoding="utf-8"))
cfg = json.load(open(cfg_path, encoding="utf-8"))
dense_k = scan.get("dense_k_by_prefix") or {}

if "mtp.fc" in dense_k:
    mtp_fc_bits = int(dense_k["mtp.fc"])
    for container in (cfg, cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else None):
        if not isinstance(container, dict):
            continue
        q = container.get("quantization_config")
        if isinstance(q, dict) and isinstance(q.get("non_routed_exl3"), dict):
            nr = q["non_routed_exl3"]
            layers = nr.setdefault("layers", {})
            layers.setdefault("mtp.fc", {"bits": mtp_fc_bits})
            layers.setdefault("model.fc", {"bits": mtp_fc_bits})
            mods = set(nr.get("modules") or [])
            mods.update({"mtp.fc", "model.fc"})
            nr["modules"] = sorted(mods)

with open(cfg_path, "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, indent=2)
    fh.write("\n")
PY

echo "3/3 regenerating $TARGET_DIR/model.safetensors.index.json ..."
python3 "$TOOLS/regenerate_safetensors_index.py" "$TARGET_DIR"

echo "pack prepared: $TARGET_DIR"
