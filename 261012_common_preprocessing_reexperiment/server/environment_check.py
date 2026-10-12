"""Check the supplied dialect1 package profile and actual Trainer construction."""

import argparse
from datetime import datetime, timezone
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import traceback


EXPECTED = {
    "torch": "2.0.1",
    "transformers": "4.40.1",
    "datasets": "2.19.1",
    "tokenizers": "0.19.1",
    "numpy": "1.26.4",
    "pyarrow": "15.0.2",
    "accelerate": "1.14.0",
    "scikit-learn": "1.4.2",
}
IMPORTS = {
    "torch": "torch", "transformers": "transformers", "datasets": "datasets",
    "tokenizers": "tokenizers", "numpy": "numpy", "pyarrow": "pyarrow",
    "accelerate": "accelerate", "scikit-learn": "sklearn",
    "sentencepiece": "sentencepiece", "tqdm": "tqdm", "psutil": "psutil",
    "nvidia-ml-py": "pynvml", "huggingface-hub": "huggingface_hub",
}


def check_environment(cpu=False, metadata_only=False, probe_dir=None, stage="region"):
    report = {
        "completed": False,
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "cpu_requested": cpu,
        "metadata_only": metadata_only, "packages": {}, "errors": [], "warnings": [],
        "stage": stage,
        "trainer_probe_passed": False, "training_smoke_passed": False,
    }
    if sys.version_info[:2] != (3, 10):
        report["errors"].append("This supplied environment profile requires Python 3.10.")
    imports = dict(IMPORTS)
    expected_versions = dict(EXPECTED)
    if stage in ("translation", "all"):
        imports.update({"sacrebleu": "sacrebleu", "rapidfuzz": "rapidfuzz"})
        expected_versions.update({"sacrebleu": "2.6.0", "rapidfuzz": "3.14.5"})
    for distribution, module in imports.items():
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            report["packages"][distribution] = {"installed": False}
            report["errors"].append(f"Missing required package: {distribution}")
            continue
        record = {"installed": True, "version": version}
        report["packages"][distribution] = record
        expected = expected_versions.get(distribution)
        if expected and version.split("+")[0] != expected:
            report["errors"].append(f"{distribution} version {version} differs from profile {expected}.")
        if not metadata_only:
            try:
                importlib.import_module(module)
                record["import_passed"] = True
            except Exception as exc:
                record["import_passed"] = False
                record["error"] = f"{type(exc).__name__}: {exc}"
                report["errors"].append(f"Import failed for {distribution}: {exc}")
    if metadata_only:
        report["warnings"].append("Metadata inspection does not verify imports, CUDA, Trainer, or training.")
        report["completed"] = not report["errors"]
        return report
    if report["errors"]:
        return report
    try:
        import torch
        from transformers import BertConfig, BertForSequenceClassification, Trainer, TrainingArguments
        cuda_available = torch.cuda.is_available()
        report["torch_cuda_build"] = torch.version.cuda
        report["cuda_available"] = cuda_available
        if not cpu and not cuda_available:
            raise RuntimeError("CUDA is unavailable. Use --cpu only for a separate CPU functional check.")
        if not cpu:
            report["cuda_device"] = torch.cuda.get_device_name(0)
            report["cuda_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
            if torch.version.cuda != "11.7":
                report["warnings"].append("CUDA build differs from the supplied torch 2.0.1+cu117 environment.")
        probe_dir = Path(probe_dir or Path(__file__).resolve().parent / "environment_checks" / "trainer_probe")
        probe_dir.mkdir(parents=True, exist_ok=True)
        config = BertConfig(vocab_size=32, hidden_size=24, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=48, num_labels=5)
        args = TrainingArguments(
            output_dir=str(probe_dir), report_to="none", no_cuda=cpu,
            fp16=not cpu, per_device_train_batch_size=1,
            optim="adamw_torch" if cpu else "adamw_torch_fused",
        )
        trainer = Trainer(model=BertForSequenceClassification(config), args=args)
        trainer.create_optimizer()
        report["trainer_probe_passed"] = True
        report["trainer_device"] = str(trainer.args.device)
        report["optimizer_class"] = type(trainer.optimizer).__name__
        report["warnings"].append("Trainer construction passed; run smoke_test.py to verify training and generation.")
        del trainer
        report["completed"] = True
    except Exception as exc:
        report["errors"].append(f"Runtime probe failed: {type(exc).__name__}: {exc}")
        report["traceback"] = traceback.format_exc()
    return report


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="Check a CPU functional environment, not server CUDA.")
    parser.add_argument("--metadata-only", action="store_true", help="Read package metadata without importing ML packages.")
    parser.add_argument("--report", type=Path, help="Save the detailed result as JSON.")
    parser.add_argument("--stage", choices=("region", "translation", "all"), default="region",
                        help="Check dialect1, dialect_translate, or an explicitly combined environment.")
    args = parser.parse_args(argv)
    if args.cpu:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    report = check_environment(cpu=args.cpu, metadata_only=args.metadata_only,
                               probe_dir=args.report.parent / "environment_probe" if args.report else None,
                               stage=args.stage)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for package, record in report["packages"].items():
        print(f"[PACKAGE] {package}: {record.get('version', 'MISSING')}")
    for warning in report["warnings"]:
        print(f"[NOTE] {warning}")
    for error in report["errors"]:
        print(f"[FAIL] {error}", file=sys.stderr)
    if report["completed"]:
        print("[OK] Environment inspection completed; this is not an end-to-end smoke-test result.")
        return 0
    print("[FAIL] Resolve the reported environment issue before running the experiment.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
