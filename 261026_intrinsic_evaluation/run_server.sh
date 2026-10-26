#!/usr/bin/env bash
# CPU-only tokenizer evaluation. All writes stay inside this project directory.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_SERVER_ROOT="/data/ohdg/kodialect-bert/261012_common_preprocessing_reexperiment/server"
MODE="full"
MODE_SET=0
BASE_PYTHON=""
OUTPUT_DIR=""
BATCH_SIZE=1024
SMOKE_LINES=1000

usage() {
    cat <<'USAGE'
Usage: bash run_server.sh [--setup|--check|--smoke|--full] [options]

  --setup                 Create project .env (Python 3.10), install pinned CPU dependencies, then stop.
  --check                 Prepare exact tokenizers, check versions/corpus path/vocabulary sizes, then stop.
  --smoke                 Evaluate the first 1000 Test rows; never publish as full results.
  --full                  Evaluate all 1,111,735 Test rows (default).
  --source-server-root DIR Existing 261012.../server directory; source inputs are read only.
  --python PATH           Base Python 3.10 used only to create .env during --setup.
  --output-dir DIR        New empty output directory for smoke/full evaluation.
  --batch-size N          Tokenization batch size (default 1024).
  --smoke-lines N         Prefix row count in --smoke mode (default 1000).
USAGE
}

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

while (($#)); do
    case "$1" in
        --setup|--check|--smoke|--full)
            ((MODE_SET == 0)) || fail 'Choose only one execution mode.'
            MODE="${1#--}"
            MODE_SET=1
            shift
            ;;
        --source-server-root|--python|--output-dir|--batch-size|--smoke-lines)
            (($# >= 2)) || fail "Missing value for $1"
            case "$1" in
                --source-server-root) SOURCE_SERVER_ROOT="$2" ;;
                --python) BASE_PYTHON="$2" ;;
                --output-dir) OUTPUT_DIR="$2" ;;
                --batch-size) BATCH_SIZE="$2" ;;
                --smoke-lines) SMOKE_LINES="$2" ;;
            esac
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown argument: $1" ;;
    esac
done

[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || fail '--batch-size must be a positive integer.'
[[ "$SMOKE_LINES" =~ ^[1-9][0-9]*$ ]] || fail '--smoke-lines must be a positive integer.'
[[ "$MODE" == setup || -z "$BASE_PYTHON" ]] || fail '--python is only used with --setup; later modes always use the project .env.'

cd -- "$PROJECT_DIR"
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"
export PIP_REQUIRE_VIRTUALENV=true
export PYTHONNOUSERSITE=1
export PYTHONUTF8=1
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_MODULES_CACHE="$PROJECT_DIR/.cache/huggingface/modules"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=""
unset PYTHONPATH PYTHONHOME || true
ENV_PYTHON="$PROJECT_DIR/.env/bin/python"

is_python310() {
    "$1" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 10))' >/dev/null 2>&1
}

if [[ "$MODE" == setup ]]; then
    if [[ ! -x "$ENV_PYTHON" ]]; then
        if [[ -n "$BASE_PYTHON" ]]; then
            [[ -x "$BASE_PYTHON" ]] || fail "Base Python is not executable: $BASE_PYTHON"
            is_python310 "$BASE_PYTHON" || fail 'The requested base Python must be version 3.10.'
        else
            for candidate in /data/ohdg/anaconda3/envs/dialect1/bin/python \
                             "$HOME/anaconda3/envs/dialect1/bin/python"; do
                if [[ -x "$candidate" ]] && is_python310 "$candidate"; then
                    BASE_PYTHON="$candidate"
                    break
                fi
            done
            if [[ -z "$BASE_PYTHON" ]] && command -v python3.10 >/dev/null 2>&1; then
                BASE_PYTHON="$(command -v python3.10)"
            fi
            [[ -n "$BASE_PYTHON" ]] || fail 'Python 3.10 was not found. Supply --python /path/to/python3.10. No system packages are installed.'
        fi
        is_python310 "$BASE_PYTHON" || fail 'The selected base interpreter must be Python 3.10.'
        printf '[SETUP] Base interpreter (read only): %s\n' "$BASE_PYTHON"
        "$BASE_PYTHON" -m venv "$PROJECT_DIR/.env"
    fi
    is_python310 "$ENV_PYTHON" || fail 'Existing project .env is not Python 3.10.'
    "$ENV_PYTHON" -m pip install --require-virtualenv -r "$PROJECT_DIR/requirements.txt"
    printf '[OK] Project environment ready: %s\n' "$ENV_PYTHON"
    printf 'Next: bash run_server.sh --check\n'
    exit 0
fi

[[ -x "$ENV_PYTHON" ]] || fail 'Project .env is missing. Run bash run_server.sh --setup first.'
is_python310 "$ENV_PYTHON" || fail 'Project .env must use Python 3.10.'
CORPUS="$SOURCE_SERVER_ROOT/region_classification/data/corpus/dialect_test_corpus.txt"
MODELS_ROOT="$SOURCE_SERVER_ROOT/region_classification/outputs/mlm"
[[ -f "$CORPUS" ]] || fail "Prepared Test corpus not found: $CORPUS"
[[ -d "$MODELS_ROOT" ]] || fail "Saved MLM directory not found: $MODELS_ROOT"
[[ -f "$PROJECT_DIR/tools/prepare_server_inputs.py" ]] || fail 'tools/prepare_server_inputs.py is missing; update the complete project code.'

"$ENV_PYTHON" "$PROJECT_DIR/tools/prepare_server_inputs.py" \
    --source-root "$MODELS_ROOT" --output-dir "$PROJECT_DIR/inputs/tokenizers"
"$ENV_PYTHON" "$PROJECT_DIR/tools/check_environment.py" \
    --corpus "$CORPUS" --tokenizers-root "$PROJECT_DIR/inputs/tokenizers"

if [[ "$MODE" == check ]]; then
    exit 0
fi

ARGS=("$PROJECT_DIR/evaluate.py" --corpus "$CORPUS" \
      --tokenizers-root "$PROJECT_DIR/inputs/tokenizers" --batch-size "$BATCH_SIZE")
if [[ "$MODE" == smoke ]]; then
    ARGS+=(--max-lines "$SMOKE_LINES")
fi
if [[ -n "$OUTPUT_DIR" ]]; then
    ARGS+=(--output-dir "$OUTPUT_DIR")
fi
exec "$ENV_PYTHON" -u "${ARGS[@]}"

