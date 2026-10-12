"""Run both original experiments, optionally using separate Python environments."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PACKAGE_NAMES = (
    "torch", "transformers", "datasets", "tokenizers", "numpy", "pyarrow",
    "accelerate", "scikit-learn", "sentencepiece", "sacrebleu", "rapidfuzz",
    "tqdm", "psutil", "nvidia-ml-py", "huggingface-hub",
)
PROBE_CODE = """import json, sys
from importlib import metadata
names = %r
versions = {}
for name in names:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps({"executable": sys.executable, "version": sys.version,
                  "prefix": sys.prefix, "base_prefix": sys.base_prefix,
                  "packages": versions}))
""" % (PACKAGE_NAMES,)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def write_record(path, record):
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_interpreter(requested):
    """Validate an executable path without dereferencing a virtualenv symlink."""
    value = os.path.expanduser(str(requested))
    if not value.strip():
        raise ValueError("Python executable must not be blank.")
    # A bare command may be resolved through PATH. A shell command such as
    # 'conda run -n dialect1 python' is not an executable and is rejected.
    has_separator = any(separator in value for separator in ("/", "\\"))
    if os.path.isabs(value) or has_separator:
        candidate = value
    else:
        candidate = shutil.which(value)
        if candidate is None:
            raise FileNotFoundError(f"Python executable was not found on PATH: {requested}")
    launch_path = Path(os.path.abspath(candidate))
    if not launch_path.is_file():
        raise FileNotFoundError(f"Python executable is not a file: {launch_path}")
    if not os.access(launch_path, os.X_OK):
        raise PermissionError(f"Python executable is not executable: {launch_path}")
    if os.name == "nt" and launch_path.suffix.lower() not in (".exe", ".com"):
        raise ValueError("On Windows pass the environment's python.exe, not a shell/batch launcher.")
    return {
        "requested": str(requested), "launch_path": str(launch_path),
        "real_path": str(launch_path.resolve()),
    }


def planned_steps(interpreters, check_only):
    specifications = [
        ("verify_bundle", "region", "verify_bundle.py", []),
        ("check_region", "region", "region_classification/run_server.py", ["--check-only"]),
        ("check_translation", "translation", "translation/run_server.py", ["--check-only"]),
    ]
    if not check_only:
        specifications.extend([
            ("train_region", "region", "region_classification/run_server.py", []),
            ("analyze_region", "region", "region_classification/run_server.py", ["--analyze-only"]),
            ("train_translation", "translation", "translation/run_server.py", []),
        ])
    return [{
        "stage": stage, "environment": role, "status": "not_started",
        "command": [interpreters[role]["launch_path"], str(ROOT / script), *arguments],
        "cwd": str(ROOT),
    } for stage, role, script, arguments in specifications]


def create_record(interpreters, check_only):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    record_dir = ROOT / "run_records" / run_id
    record_dir.mkdir(parents=True, exist_ok=False)
    code_paths = {
        "workflow": ROOT / "run_all.py",
        "region_wrapper": ROOT / "region_classification/run_server.py",
        "region_original_runner": ROOT / "region_classification/run_full_experiment.py",
        "translation_wrapper": ROOT / "translation/run_server.py",
        "translation_original_runner": ROOT / "translation/run_full_experiment.py",
        "original_code_index": ROOT / "original_code_sha256.json",
    }
    code_records = {
        name: {"path": str(path), "sha256": file_sha256(path) if path.is_file() else None}
        for name, path in code_paths.items()
    }
    record = {
        "run_id": run_id, "started_utc": utc_now(), "status": "running", "completed": False,
        "mode": "check_only" if check_only else "full_workflow",
        "server_root": str(ROOT), "interpreters": interpreters,
        "interpreter_checks": [], "steps": planned_steps(interpreters, check_only),
        "code": code_records,
        "source_mlm_root": str(ROOT / "region_classification/outputs/mlm"),
        "source_mlm_model_paths": {
            tokenizer: str(ROOT / "region_classification/outputs/mlm" / tokenizer / "final_model")
            for tokenizer in ("dialect", "klue", "kobert", "mbert")
        },
        "shared_decoder_initial_model": str(ROOT / "translation/outputs/shared_standard_decoder/final_model"),
        "provenance_note": (
            "The preserved translation summary template mentions the old 260807 experiment. "
            "That historical sentence does not describe this run: translation receives the explicit "
            "source_mlm_root recorded above, from this bundle's region experiment. "
            "No preserved training or metric code is edited."
        ),
        "environment_note": "Package versions are inspected and recorded; no package is installed or changed.",
    }
    path = record_dir / "run_provenance.json"
    write_record(path, record)
    return path, record


def probe_interpreter(role, info, env, record_path, record):
    event = {
        "environment": role, "status": "running", "started_utc": utc_now(),
        "command": [info["launch_path"], "-c", PROBE_CODE], "cwd": str(ROOT),
    }
    record["interpreter_checks"].append(event)
    write_record(record_path, record)
    try:
        result = subprocess.run(
            event["command"], cwd=ROOT, env=env, capture_output=True,
            text=True, encoding="utf-8", errors="replace", check=False,
        )
        event["returncode"] = result.returncode
        if result.returncode:
            event["stderr"] = result.stderr[-4000:]
            raise RuntimeError(f"{role} Python probe failed (exit {result.returncode}): {result.stderr.strip()}")
        try:
            runtime = json.loads(result.stdout)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{role} executable did not return a valid Python environment report.") from exc
        if not isinstance(runtime, dict) or not isinstance(runtime.get("executable"), str):
            raise RuntimeError(f"{role} Python environment report is incomplete.")
        info["runtime"] = runtime
        event["status"] = "completed"
        print(f"[PYTHON {role}] {runtime['executable']}", flush=True)
    except BaseException as exc:
        event["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        event["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        event["finished_utc"] = utc_now()
        write_record(record_path, record)


def execute_step(step, env, record_path, record):
    step.update({"status": "running", "started_utc": utc_now()})
    write_record(record_path, record)
    print(f"[RUN {step['environment']}] {step['stage']}", flush=True)
    try:
        result = subprocess.run(step["command"], cwd=ROOT, env=env, check=False)
        step["returncode"] = result.returncode
        if result.returncode:
            raise RuntimeError(f"Stage {step['stage']} failed with exit code {result.returncode}.")
        step["status"] = "completed"
    except BaseException as exc:
        step["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        step["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        step["finished_utc"] = utc_now()
        write_record(record_path, record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Validate the prepared bundle; do not train.")
    parser.add_argument("--region-python", default=sys.executable, help="Region environment's Python executable path (e.g. dialect1/bin/python); defaults to this Python.")
    parser.add_argument("--translation-python", default=sys.executable, help="Translation environment's Python executable path (e.g. dialect_translate/bin/python); defaults to this Python.")
    args = parser.parse_args(argv)
    interpreters = {
        "region": resolve_interpreter(args.region_python),
        "translation": resolve_interpreter(args.translation_python),
    }
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    record_path, record = create_record(interpreters, args.check_only)
    print(f"[PROVENANCE] {record_path}", flush=True)
    try:
        for role, info in interpreters.items():
            probe_interpreter(role, info, env, record_path, record)
        for step in record["steps"]:
            execute_step(step, env, record_path, record)
        record.update({"status": "completed", "completed": True})
        return 0
    except KeyboardInterrupt:
        record.update({"status": "interrupted", "error": "KeyboardInterrupt"})
        raise
    except Exception as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        record["finished_utc"] = utc_now()
        write_record(record_path, record)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
