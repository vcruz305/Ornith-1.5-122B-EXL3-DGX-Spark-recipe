#!/usr/bin/env bash
#
# setup_env.sh: Create a working Python environment for serving
# vcruz305/Ornith-1.5-122B-EXL3-SAGE-2.58bpw through vLLM 0.32.0 + vllm-exl3.
#
# Steps performed:
#   1. Verify Python >= 3.10 and CUDA / host toolchain preconditions.
#   2. Create or reuse (--force) the virtual environment at $VENV.
#   3. Install vllm==0.32.0 (pinned exact version).
#   4. Build or link vcruz305/exllamav3 (exllamav3_ext with exl3_moe_coop, exl3_moe, exl3_mgemm).
#   5. Install vcruz305/vllm-exl3 with NVCC_APPEND_FLAGS="-DP2B_CB=2" (mul1 codebook).
#   6. Apply patch_vllm_qwen3_5_vision_split.py to vLLM's Qwen3_VisionTransformer.
#   7. Run vllm-plugin/preflight.py to verify all components.
#
# TARGET PLATFORM: Linux aarch64 (NVIDIA DGX Spark, GB10, sm_121).
#

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

VENV="${VENV:-$HOME/venvs/vllm-exl3}"
VLLM_VERSION="${VLLM_VERSION:-0.32.0}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1}"
NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:--DP2B_CB=2}"

PLUGIN_SOURCE="git"
PLUGIN_REF="${PLUGIN_REF:-main}"
EXL3_REPO="${EXL3_REPO:-https://github.com/vcruz305/exllamav3.git}"
EXL3_REF="${EXL3_REF:-master}"
EXL3_SRC="${EXL3_SRC:-}"
LINK_EXL3_VENV="${LINK_EXL3_VENV:-}"
PACK_DIR="${PACK_DIR:-}"

FORCE_VENV=0
SKIP_PATCHES=0
DID_CREATE_VENV=0

cleanup() {
    if [[ $? -ne 0 ]] && [[ $DID_CREATE_VENV -eq 1 ]]; then
        printf "%b\n" "${YELLOW}Setup failed. Venv was created at ${VENV}; remove it manually or retry with --force.${NC}"
    fi
}
trap cleanup EXIT

show_help() {
    cat <<EOF
usage: bash vllm-plugin/setup_env.sh [OPTIONS]

Create a working Python environment for serving vcruz305/Ornith-1.5-122B-EXL3-SAGE-2.58bpw
through vLLM ${VLLM_VERSION} + vcruz305/vllm-exl3.

OPTIONS:
  --venv PATH              Path to venv directory (default: \$HOME/venvs/vllm-exl3)
  --plugin-src PATH        Install vllm-exl3 from local source directory (pip install --no-build-isolation -e)
  --plugin-git REF         Install vllm-exl3 from git repository at given ref (default: main)
  --exl3-src PATH          Build/install vcruz305/exllamav3 from local checkout (or clone there)
  --link-exl3-venv PATH    Symlink prebuilt exllamav3 + exllamav3_ext from an ABI-matching venv
  --pack PATH              Optional prepared pack path to pass to preflight.py
  --force                  Overwrite existing venv instead of refusing
  --skip-patches           Do not apply patch_vllm_qwen3_5_vision_split.py after installation
  -h, --help               Show this help message

ENVIRONMENT:
  VENV                     Override default venv path
  VLLM_VERSION             Pin vLLM version (default: 0.32.0)
  NVCC_APPEND_FLAGS        Compiler flags for vllm-exl3 (default: -DP2B_CB=2 for mul1 codebook)
  TORCH_CUDA_ARCH_LIST     Target CUDA arch for source builds (default: 12.1)
  CUDA_HOME                CUDA toolkit root (default: /usr/local/cuda)
  EXL3_SRC                 Checkout path for vcruz305/exllamav3
  LINK_EXL3_VENV           Existing venv containing ABI-matching exllamav3_ext
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)
            VENV="$2"
            shift 2
            ;;
        --plugin-src)
            PLUGIN_SOURCE="src"
            PLUGIN_REF="$2"
            shift 2
            ;;
        --plugin-git)
            PLUGIN_SOURCE="git"
            PLUGIN_REF="$2"
            shift 2
            ;;
        --exl3-src)
            EXL3_SRC="$2"
            shift 2
            ;;
        --link-exl3-venv)
            LINK_EXL3_VENV="$2"
            shift 2
            ;;
        --pack)
            PACK_DIR="$2"
            shift 2
            ;;
        --force)
            FORCE_VENV=1
            shift
            ;;
        --skip-patches)
            SKIP_PATCHES=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            printf "%b\n" "${RED}error: unknown option '$1'${NC}"
            show_help
            exit 1
            ;;
    esac
done

log_step() {
    printf "%b\n" "${BLUE}[Step]${NC} $1"
}

log_info() {
    printf "%b\n" "${GREEN}[INFO]${NC} $1"
}

log_error() {
    printf "%b\n" "${RED}[ERROR]${NC} $1"
}

log_command() {
    printf "%b\n" "${YELLOW}[RUN]${NC} $1"
}

fail() {
    log_error "$1"
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Step 1: Preconditions
log_step "Checking preconditions"

if ! command -v python3 &>/dev/null; then
    fail "python3 not found in PATH"
fi

PYTHON3_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
PYTHON3_MAJOR=$(python3 -c 'import sys; print(sys.version_info[0])')
PYTHON3_MINOR=$(python3 -c 'import sys; print(sys.version_info[1])')

if [[ $PYTHON3_MAJOR -lt 3 ]] || { [[ $PYTHON3_MAJOR -eq 3 ]] && [[ $PYTHON3_MINOR -lt 10 ]]; }; then
    fail "python3 must be >= 3.10 (found $PYTHON3_VERSION)"
fi

log_info "python3 version: $PYTHON3_VERSION"
log_info "architecture: $(uname -m)"
log_info "operating system: $(uname -s)"

if [[ -d "$CUDA_HOME/bin" ]]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export CUDA_HOME
fi

# Step 2: Prepare or create venv
log_step "Preparing virtual environment"

if [[ -d "$VENV" ]]; then
    if [[ $FORCE_VENV -eq 0 ]]; then
        printf "%b\n" "${YELLOW}[REFUSE]${NC} Virtual environment already exists at:"
        printf "  %s\n" "$VENV"
        printf "\nTo proceed, either:\n"
        printf "  1. Remove the directory manually: rm -rf %s\n" "$VENV"
        printf "  2. Use --force flag to overwrite it\n"
        exit 1
    else
        log_info "Overwriting existing venv at $VENV (--force was used)"
        log_command "rm -rf '$VENV'"
        rm -rf "$VENV"
    fi
fi

log_command "python3 -m venv '$VENV'"
python3 -m venv "$VENV" || fail "Failed to create venv at $VENV"
DID_CREATE_VENV=1

# shellcheck disable=SC1090,SC1091
source "$VENV/bin/activate" || fail "Failed to activate venv"
export PATH="$VENV/bin:$PATH"
log_info "Virtual environment activated: $VENV"

log_command "pip install --upgrade pip setuptools wheel ninja packaging"
pip install --upgrade pip setuptools wheel ninja packaging || fail "Failed to upgrade build tools"

# Step 3: Install vLLM
log_step "Installing vLLM version ${VLLM_VERSION}"

log_command "pip install 'vllm==${VLLM_VERSION}'"
pip install "vllm==${VLLM_VERSION}" || fail "Failed to install vllm==${VLLM_VERSION}"

INSTALLED_VLLM=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
log_info "Installed vLLM version: $INSTALLED_VLLM"

INSTALLED_TORCH=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
log_info "Installed PyTorch version: $INSTALLED_TORCH"

# Step 4: Build or link vcruz305/exllamav3 (exllamav3_ext)
log_step "Setting up vcruz305/exllamav3 (exllamav3_ext)"

check_exl3_ext() {
    python3 -c "import torch, exllamav3_ext; assert all(hasattr(exllamav3_ext, k) for k in ('exl3_moe_coop', 'exl3_moe', 'exl3_mgemm'))" 2>/dev/null
}

if [[ -n "$LINK_EXL3_VENV" ]]; then
    if [[ ! -d "$LINK_EXL3_VENV" ]]; then
        fail "LINK_EXL3_VENV does not exist: $LINK_EXL3_VENV"
    fi
    log_info "Linking exllamav3 / exllamav3_ext from $LINK_EXL3_VENV"
    python3 - "$LINK_EXL3_VENV" "$VENV" <<'PY'
import glob
import os
import sys

src_venv, dst_venv = sys.argv[1], sys.argv[2]
src_sites = glob.glob(os.path.join(src_venv, "lib", "python*", "site-packages"))
dst_sites = glob.glob(os.path.join(dst_venv, "lib", "python*", "site-packages"))
if not src_sites or not dst_sites:
    raise SystemExit(f"Could not resolve site-packages between {src_venv} and {dst_venv}")
src_site, dst_site = src_sites[0], dst_sites[0]

linked = []
for pattern in ("exllamav3_ext*.so", "exllamav3", "exllamav3-*.dist-info", "__editable__.exllamav3*.pth", "_exllamav3*.pth"):
    for item in glob.glob(os.path.join(src_site, pattern)):
        base = os.path.basename(item)
        target = os.path.join(dst_site, base)
        if os.path.lexists(target):
            os.unlink(target)
        os.symlink(os.path.realpath(item), target)
        linked.append(base)
if not linked:
    raise SystemExit(f"No exllamav3 artifacts found in {src_site}")
print("Linked artifacts:", ", ".join(linked))
PY
fi

if ! check_exl3_ext; then
    if [[ -z "$EXL3_SRC" ]]; then
        for cand in "$HOME/ornith-exl3/exllamav3" "$HOME/qwen38-exl3/exllamav3" "$(dirname "$VENV")/exllamav3"; do
            if [[ -d "$cand/.git" ]]; then
                EXL3_SRC="$cand"
                break
            fi
        done
        EXL3_SRC="${EXL3_SRC:-$(dirname "$VENV")/exllamav3}"
    fi
    if [[ ! -d "$EXL3_SRC/.git" ]]; then
        log_command "git clone '$EXL3_REPO' '$EXL3_SRC'"
        git clone "$EXL3_REPO" "$EXL3_SRC" || fail "Failed to clone $EXL3_REPO"
        git -C "$EXL3_SRC" checkout "$EXL3_REF" || fail "Failed to checkout $EXL3_REF"
    fi
    log_info "Building vcruz305/exllamav3 from $EXL3_SRC (TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST)"
    log_command "TORCH_CUDA_ARCH_LIST='$TORCH_CUDA_ARCH_LIST' pip install --no-build-isolation -e '$EXL3_SRC'"
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" MAX_JOBS="${MAX_JOBS:-$(nproc 2>/dev/null || echo 4)}" \
        pip install --no-build-isolation -e "$EXL3_SRC" || fail "Failed to build exllamav3 from $EXL3_SRC"
fi

if check_exl3_ext; then
    log_info "exllamav3_ext (exl3_moe_coop, exl3_moe, exl3_mgemm) verified. PASS"
else
    fail "exllamav3_ext is missing required kernels (exl3_moe_coop, exl3_moe, exl3_mgemm)"
fi

# Step 5: Install vcruz305/vllm-exl3 with NVCC_APPEND_FLAGS="-DP2B_CB=2"
log_step "Installing vcruz305/vllm-exl3 plugin (NVCC_APPEND_FLAGS='${NVCC_APPEND_FLAGS}')"
export NVCC_APPEND_FLAGS
export TORCH_CUDA_ARCH_LIST

if [[ "$PLUGIN_SOURCE" == "src" ]]; then
    if [[ ! -d "$PLUGIN_REF" ]]; then
        fail "Plugin source directory does not exist: $PLUGIN_REF"
    fi
    log_command "NVCC_APPEND_FLAGS='$NVCC_APPEND_FLAGS' pip install --no-build-isolation -e '$PLUGIN_REF'"
    pip install --no-build-isolation -e "$PLUGIN_REF" || fail "Failed to install vllm-exl3 from $PLUGIN_REF"
    log_info "Installed vllm-exl3 from local source: $PLUGIN_REF"
else
    log_command "NVCC_APPEND_FLAGS='$NVCC_APPEND_FLAGS' pip install --no-build-isolation 'git+https://github.com/vcruz305/vllm-exl3@${PLUGIN_REF}'"
    pip install --no-build-isolation "git+https://github.com/vcruz305/vllm-exl3@${PLUGIN_REF}" || fail "Failed to install vllm-exl3 from git ref $PLUGIN_REF"
    log_info "Installed vllm-exl3 from git ref: $PLUGIN_REF"
fi

# Step 6: Apply vLLM Qwen3.5 vision split patch
if [[ $SKIP_PATCHES -eq 0 ]]; then
    log_step "Applying patch_vllm_qwen3_5_vision_split.py"
    PATCH_SCRIPT="$SCRIPT_DIR/patch_vllm_qwen3_5_vision_split.py"
    if [[ ! -f "$PATCH_SCRIPT" ]] && [[ "$PLUGIN_SOURCE" == "src" ]]; then
        for cand in \
            "$PLUGIN_REF/tools/patch_vllm_qwen3_5_vision_split.py" \
            "$PLUGIN_REF/tools/patch_vllm_qwen4_exp/patch_vllm_qwen3_5_vision_split.py"; do
            [[ -f "$cand" ]] && { PATCH_SCRIPT="$cand"; break; }
        done
    fi
    if [[ ! -f "$PATCH_SCRIPT" ]]; then
        fail "Could not locate patch_vllm_qwen3_5_vision_split.py"
    fi
    VLLM_LIB=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null || echo "")
    if [[ -z "$VLLM_LIB" ]]; then
        fail "Could not locate vllm installation directory"
    fi
    log_command "python3 '$PATCH_SCRIPT' '$VLLM_LIB'"
    python3 "$PATCH_SCRIPT" "$VLLM_LIB" || fail "patch_vllm_qwen3_5_vision_split.py failed"
    log_info "Vision split patch applied"
else
    log_step "Skipping vLLM patches (--skip-patches)"
fi

# Step 7: Run preflight checks
log_step "Running preflight checks"

PREFLIGHT_SCRIPT="$SCRIPT_DIR/preflight.py"
if [[ -f "$PREFLIGHT_SCRIPT" ]]; then
    PREFLIGHT_ARGS=()
    if [[ -n "$PACK_DIR" ]]; then
        PREFLIGHT_ARGS+=(--pack "$PACK_DIR")
    fi
    log_command "python3 '$PREFLIGHT_SCRIPT' ${PREFLIGHT_ARGS[*]}"
    python3 "$PREFLIGHT_SCRIPT" "${PREFLIGHT_ARGS[@]}" || fail "Preflight check failed"
else
    log_info "No preflight.py found; skipping preflight checks"
fi

cat <<EOF

${GREEN}+====================================================================+${NC}
${GREEN}| Setup complete                                                     |${NC}
${GREEN}+====================================================================+${NC}
${GREEN}| Virtual environment path:                                          |${NC}
${GREEN}|   ${NC}$VENV
${GREEN}|                                                                    |${NC}
${GREEN}| To activate in your current shell:                                 |${NC}
${GREEN}|   ${NC}source $VENV/bin/activate
${GREEN}|                                                                    |${NC}
${GREEN}| Next steps:                                                        |${NC}
${GREEN}|   1. Prepare pack view: bash vllm-plugin/prepare_pack.sh           |${NC}
${GREEN}|   2. Serve:             bash vllm-plugin/serve.sh                  |${NC}
${GREEN}+====================================================================+${NC}

EOF

log_info "All setup steps completed successfully"
