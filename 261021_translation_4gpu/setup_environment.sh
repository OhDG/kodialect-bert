#!/usr/bin/env bash
# Linux x86_64 only. No sudo, driver changes, shell init, or base-env pip installs.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PREFIX=""
PYTHON_BIN=""
BOOTSTRAP=0
MODE=auto
usage() {
  cat <<'HELP'
Usage: bash setup_environment.sh [--bootstrap-conda] [--venv] [--python /path/python3.10] [--prefix /new/env]

Default: use an available conda to create .env; otherwise use Python 3.10
to create .venv. Existing project environments are reusable. --venv forces
venv. --bootstrap-conda explicitly downloads and verifies Miniforge when
conda is absent; installation stays inside this bundle's .miniforge folder.
The NVIDIA driver must already be installed by the server administrator.
HELP
}
while (($#)); do
  case "$1" in
    --bootstrap-conda) BOOTSTRAP=1; shift ;;
    --venv) MODE=venv; shift ;;
    --python) PYTHON_BIN="${2:?--python requires a path}"; MODE=venv; shift 2 ;;
    --prefix) PREFIX="${2:?--prefix requires a path}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage; exit 2 ;;
  esac
done
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
  echo 'This setup targets Linux x86_64 with NVIDIA GPUs.' >&2; exit 2;
}
CONDA_BIN="${CONDA_EXE:-}"
if [[ -z "$CONDA_BIN" ]]; then CONDA_BIN="$(command -v conda || true)"; fi
if [[ -z "$CONDA_BIN" && -x "$ROOT/.miniforge/bin/conda" ]]; then
  CONDA_BIN="$ROOT/.miniforge/bin/conda"
fi
if [[ "$MODE" != venv && -z "$CONDA_BIN" && "$BOOTSTRAP" == 1 ]]; then
  command -v curl >/dev/null || { echo 'Install curl before bootstrap.' >&2; exit 2; }
  command -v sha256sum >/dev/null || { echo 'sha256sum is required.' >&2; exit 2; }
  [[ ! -e "$ROOT/.miniforge" ]] || {
    echo 'Incomplete .miniforge exists; inspect it before retrying.' >&2; exit 2;
  }
  # Official versioned release + published checksum, verified 2026-10-21.
  # https://github.com/conda-forge/miniforge/releases/tag/26.7.2-0
  INSTALLER=Miniforge3-26.7.2-0-Linux-x86_64.sh
  EXPECTED_SHA256=281b0ac7d550802efc81af633225a5e6116d29ae72f3ab4eae7168c3931a4c05
  BASE_URL=https://github.com/conda-forge/miniforge/releases/download/26.7.2-0
  DOWNLOAD_DIR="$(mktemp -d)"
  trap 'rm -f -- "$DOWNLOAD_DIR/$INSTALLER"; rmdir -- "$DOWNLOAD_DIR"' EXIT
  curl --fail --location --retry 3 "$BASE_URL/$INSTALLER" -o "$DOWNLOAD_DIR/$INSTALLER"
  (cd "$DOWNLOAD_DIR" && printf '%s  %s\n' "$EXPECTED_SHA256" "$INSTALLER" | sha256sum --check -)
  bash "$DOWNLOAD_DIR/$INSTALLER" -b -p "$ROOT/.miniforge"
  CONDA_BIN="$ROOT/.miniforge/bin/conda"
fi
if [[ "$MODE" != venv && -n "$CONDA_BIN" ]]; then
  PREFIX="${PREFIX:-$ROOT/.env}"
  [[ ! -e "$PREFIX" || -f "$PREFIX/conda-meta/history" ]] || {
    echo "Refusing to use non-conda directory: $PREFIX" >&2; exit 2;
  }
  # Never operate on conda base, even if explicitly passed as --prefix.
  BASE_PREFIX="$("$CONDA_BIN" info --base)"
  [[ "$(realpath -m "$PREFIX")" != "$(realpath -m "$BASE_PREFIX")" ]] || {
    echo 'Refusing to install into conda base.' >&2; exit 2;
  }
  if [[ ! -x "$PREFIX/bin/python" ]]; then
    "$CONDA_BIN" create --yes --prefix "$PREFIX" --override-channels -c conda-forge python=3.10 pip
  fi
else
  PREFIX="${PREFIX:-$ROOT/.venv}"
  [[ ! -e "$PREFIX" || -f "$PREFIX/pyvenv.cfg" ]] || {
    echo "Refusing to use a non-venv directory: $PREFIX" >&2; exit 2;
  }
  if [[ ! -x "$PREFIX/bin/python" ]]; then
    if [[ -z "$PYTHON_BIN" ]]; then
      for CANDIDATE in python3.10 python3 python; do
        if command -v "$CANDIDATE" >/dev/null && "$CANDIDATE" -c 'import sys; assert sys.version_info[:2] == (3, 10)' 2>/dev/null; then
          PYTHON_BIN="$(command -v "$CANDIDATE")"; break
        fi
      done
    fi
    if [[ -z "$PYTHON_BIN" ]]; then
      echo 'Python 3.10 or conda was not found.' >&2
      echo 'Run: bash setup_environment.sh --bootstrap-conda' >&2
      echo 'Or install Miniforge: https://github.com/conda-forge/miniforge#install' >&2
      exit 2
    fi
    "$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Python 3.10 is required"'
    "$PYTHON_BIN" -m venv "$PREFIX"
  fi
fi
ENV_PYTHON="$PREFIX/bin/python"
"$ENV_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Existing environment must use Python 3.10"'
"$ENV_PYTHON" -m pip install 'torch==2.0.1+cu118' --index-url https://download.pytorch.org/whl/cu118
"$ENV_PYTHON" -m pip install -r "$ROOT/requirements.txt"
"$ENV_PYTHON" -m pip check
"$ENV_PYTHON" -m pip freeze > "$ROOT/environment.freeze.txt"
"$ENV_PYTHON" "$ROOT/tools/smoke_test.py" --check-environment --output-dir "$ROOT/smoke/environment"
printf '\nEnvironment ready: %s\n' "$ENV_PYTHON"
printf 'Run GPU smoke: "%s" "%s/tools/smoke_test.py" --gpus 0 1 2 3\n' "$ENV_PYTHON" "$ROOT"
