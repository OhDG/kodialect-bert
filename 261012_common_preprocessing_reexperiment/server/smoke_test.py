"""Execute the real experiment modules on an isolated synthetic smoke corpus.

Default: CUDA, original 6x384 BERT shapes, four real tokenizer conditions,
two optimizer steps per training stage, and one downstream seed. The synthetic
KRD tokenizer has 128 tokens and is NEVER a replacement for the production
32,000-token tokenizer. All outputs live beneath server/smoke_runs/.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import unicodedata
import uuid


ROOT = Path(__file__).resolve().parent
SMOKE_ROOT = ROOT / "smoke_runs"
TOKENIZERS = ("dialect", "klue", "kobert", "mbert")
REGIONS = ("강원도", "경상도", "전라도", "제주도", "충청도")
SEED = 13
STEPS = 2
SCRIPTS = {
    "tokenizer": ("region_classification", "step2_train_dialect_tokenizer.py"),
    "mlm": ("region_classification", "step3_pretrain_small_bert_mlm.py"),
    "classify": ("region_classification", "step4_finetune_region_classifier.py"),
    "changed": ("region_classification", "step6_changed_subset_analysis.py"),
    "region_summary": ("region_classification", "step5_summarize_results.py"),
    "decoder": ("translation", "step2_pretrain_standard_decoder_clm.py"),
    "translate": ("translation", "step3_train_dialect_to_standard.py"),
    "translation_summary": ("translation", "step4_summarize_results.py"),
}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def original_hashes():
    index = read_json(ROOT / "original_code_sha256.json")
    for relative, record in index.items():
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != record["sha256"]:
            raise RuntimeError(f"Original experiment code changed: {relative}")
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    return {key: value["sha256"] for key, value in index.items()}


def plan(selected="all"):
    stages = [("tokenizer", None)]
    for tokenizer in TOKENIZERS:
        stages.extend([("mlm", tokenizer), ("classify", tokenizer)])
    stages.extend([("changed", None), ("region_summary", None), ("decoder", None)])
    stages.extend(("translate", tokenizer) for tokenizer in TOKENIZERS)
    stages.append(("translation_summary", None))
    if selected == "region":
        return [item for item in stages if item[0] not in ("decoder", "translate", "translation_summary")]
    if selected == "translation":
        return [item for item in stages if item[0] in ("decoder", "translate", "translation_summary")]
    return stages


def make_fixture(run):
    """Synthetic train/validation/test rows, all five regions and both subsets."""
    import csv
    import numpy as np
    manifest = []
    cached = {key: [] for key in ("texts", "regions", "changed_clean", "changed_raw", "has_standard")}
    for split in ("train", "validation", "test"):
        cp = run / "region_data/region_classification" / f"dialect_region_{split}.tsv"
        cc = run / "region_data/corpus" / f"dialect_{split}_corpus.txt"
        tp = run / "translation_data/translation" / f"translation_{split}.tsv"
        tc = run / "translation_data/corpus" / f"standard_{split}_corpus.txt"
        for path in (cp, cc, tp, tc):
            path.parent.mkdir(parents=True, exist_ok=True)
        with cp.open("w", encoding="utf-8-sig", newline="") as cf, \
                cc.open("w", encoding="utf-8") as corpus, \
                tp.open("w", encoding="utf-8-sig", newline="") as pf, \
                tc.open("w", encoding="utf-8") as targets:
            cw, pw = csv.writer(cf, delimiter="\t"), csv.writer(pf, delimiter="\t")
            cw.writerow(["text", "label", "region", "file_id", "source_group", "utterance_id"])
            pw.writerow(["source_text", "target_text", "region", "file_id", "source_group", "is_changed", "utterance_id"])
            for region_id, region in enumerate(REGIONS):
                source_path = run / "synthetic_source" / f"{region_id}_{split}.json"
                file_id = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:20]
                utterances = []
                for variant in range(2):
                    source = (f"{region} 마을에서 오늘 아침에 가족 친구 이웃 사람들이 함께 이야기를 나누었어요. "
                              f"학교 병원 시장 바다 산과 들에서 사는 생활도 떠올렸어요. {region_id}{variant} "
                              '(추가 설명) [기호] ㆍ ㄱ "인용" #')
                    source = unicodedata.normalize("NFKC", source)
                    target = source if variant == 0 else source.replace("나누었어요", "나누었습니다").replace("떠올렸어요", "떠올렸습니다")
                    changed = source != target
                    utterance_id = f"{file_id}:{variant}"
                    cw.writerow([source, region_id, region, file_id, "synthetic_smoke", utterance_id])
                    corpus.write(source + "\n")
                    pw.writerow([source, target, region, file_id, "synthetic_smoke", int(changed), utterance_id])
                    targets.write(target + "\n")
                    # The raw annotation is identical on both sides, but selecting
                    # dialect/standard branches makes the canonical pair changed.
                    raw_source = source.replace("나누었어요", "(나누었어요)/(나누었습니다)").replace(
                        "떠올렸어요", "(떠올렸어요)/(떠올렸습니다)") if changed else source
                    raw_target = raw_source if changed else target
                    utterances.append({"dialect_form": raw_source, "standard_form": raw_target})
                    if split == "test":
                        for key, value in (("texts", source), ("regions", region),
                                           ("changed_clean", changed), ("changed_raw", raw_source != raw_target), ("has_standard", True)):
                            cached[key].append(value)
                save_json(source_path, {"utterance": utterances})
                manifest.append(dict(region=region, source_group="synthetic_smoke", source_type="base",
                                     path=str(source_path), num_sentences=2, split=split))
    manifest_path = run / "region_data/corpus_split_manifest_80_10_10.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    results = run / "region_results"
    results.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(results / "changed_flags_cache.npz", **{
        key: np.array(values, dtype=object if key in ("texts", "regions") else bool)
        for key, values in cached.items()
    })
    (run / "logs").mkdir(exist_ok=True)
    save_json(run / "fixture_contract.json", {
        "synthetic_only": True, "rows_per_split": 10, "changed_test_rows": 5,
        "downstream_seeds": [SEED], "optimizer_steps_per_stage": STEPS,
        "bert_shape": {"layers": 6, "hidden_size": 384, "heads": 6, "intermediate_size": 1536},
        "smoke_dialect_vocab_requested": 128, "production_dialect_vocab": 32000,
        "production_data_and_models_used": False,
        "warning": "Synthetic rows repeat across splits to test execution only; scores are not research results.",
    })


def import_original(stage):
    group, filename = SCRIPTS[stage]
    directory = ROOT / group
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("smoke_original_stage", directory / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_original(module, command):
    old = sys.argv
    sys.argv = [module.__file__, *map(str, command)]
    try:
        return module.parse_args()
    finally:
        sys.argv = old


def patch_short_training(module):
    original = module.build_training_arguments

    def smoke_arguments(args):
        result = original(args)
        result.max_steps = STEPS
        result.save_only_model = True
        result.disable_tqdm = True
        return result

    module.build_training_arguments = smoke_arguments


def short_options(cpu):
    values = ["--num_train_epochs", "1", "--train_batch_size", "2", "--eval_batch_size", "2",
              "--gradient_accumulation_steps", "1", "--dataloader_num_workers", "0",
              "--preprocessing_num_workers", "1", "--tokenize_batch_size", "16", "--logging_steps", "1"]
    if cpu:
        values += ["--no-fp16", "--no-tf32", "--optim", "adamw_torch"]
    return values


def verify_training_and_reload(stage, output):
    """Verify actual steps, checkpoint save, model reload, and finite inference."""
    import gc
    import torch
    from transformers import (AutoModelForMaskedLM, AutoModelForSequenceClassification,
                              AutoTokenizer, BertLMHeadModel, EncoderDecoderModel)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    state = read_json(output / "trainer_state.json")
    if state["global_step"] != STEPS:
        raise RuntimeError(f"{stage}: expected {STEPS} optimizer steps, got {state['global_step']}")
    checkpoints = sorted(output.glob("checkpoint-*/config.json"))
    if not checkpoints:
        raise RuntimeError(f"{stage}: no real saved model checkpoint")
    metrics = read_json(output / "train_results.json")
    loss = metrics.get("train_loss")
    if not isinstance(loss, (float, int)) or not math.isfinite(loss):
        raise RuntimeError(f"{stage}: training loss is missing or nonfinite: {loss}")
    final = output / "final_model"
    if stage == "translate":
        model = EncoderDecoderModel.from_pretrained(str(final))
        source_tokenizer = AutoTokenizer.from_pretrained(str(final / "source_tokenizer"), trust_remote_code=True)
        inputs = source_tokenizer("오늘 마을에서 이야기를 나누었습니다.", return_tensors="pt")
        model.eval()
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=4, num_beams=1)
        if generated.shape[0] != 1 or generated.shape[1] < 2:
            raise RuntimeError("Reloaded encoder-decoder did not generate a sequence")
        shape = model.encoder.config
        result = {"reloaded_generation_tokens": generated.shape[1]}
    else:
        classes = {"mlm": AutoModelForMaskedLM, "classify": AutoModelForSequenceClassification,
                   "decoder": BertLMHeadModel}
        model = classes[stage].from_pretrained(str(final))
        tokenizer = AutoTokenizer.from_pretrained(str(final), trust_remote_code=True)
        inputs = tokenizer("오늘 마을에서 이야기를 나누었습니다.", return_tensors="pt")
        model.eval()
        with torch.no_grad():
            logits = model(**inputs).logits
        if not torch.isfinite(logits).all().item():
            raise RuntimeError(f"Reloaded {stage} model produced nonfinite logits")
        shape = model.config
        result = {"reloaded_logits_shape": list(logits.shape)}
    if (shape.num_hidden_layers, shape.hidden_size, shape.num_attention_heads, shape.intermediate_size) != (6, 384, 6, 1536):
        raise RuntimeError("Smoke model no longer has the original small-BERT shape")
    result.update({"global_step": state["global_step"], "finite_train_loss": loss,
                   "saved_checkpoint": str(checkpoints[-1].parent), "model_reloaded": True})
    del model
    gc.collect()
    return result


def execute_child(stage, tokenizer, run, cpu):
    if not run.resolve().is_relative_to(SMOKE_ROOT.resolve()):
        raise RuntimeError("Smoke workdir must stay inside server/smoke_runs")
    original_hashes()
    import torch
    torch.set_num_threads(2)
    module = import_original(stage)
    run_report = read_json(run / "smoke_report.json")
    source_run = Path(run_report.get("region_run") or run)
    if not source_run.resolve().is_relative_to(SMOKE_ROOT.resolve()):
        raise RuntimeError("MLM inputs must come from an isolated smoke run, never production")
    dataset_cache = Path(run_report["dataset_cache"])
    if os.name == "nt":
        if not dataset_cache.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()) or not dataset_cache.name.startswith("kdsm_"):
            raise RuntimeError("Windows smoke dataset cache must be an isolated temporary directory")
    elif dataset_cache != run / "c":
        raise RuntimeError("Unexpected smoke dataset cache")
    region_out = source_run / "region_outputs"
    translation_out = run / "translation_outputs"
    common = short_options(cpu)
    output = None
    if stage == "tokenizer":
        args = parse_original(module, ["--corpus", run / "region_data/corpus/dialect_train_corpus.txt",
                                      "--output_dir", run / "dialect_tokenizer", "--vocab_size", "128",
                                      "--min_frequency", "1"])
        module.train_tokenizer(args)
        count = len((run / "dialect_tokenizer/vocab.txt").read_text(encoding="utf-8").splitlines())
        if count != 128:
            raise RuntimeError(f"Smoke tokenizer should learn 128 entries, got {count}")
        result = {"trained_vocab_size": count, "synthetic_smoke_only": True}
    elif stage == "mlm":
        output = region_out / "mlm" / tokenizer
        args = parse_original(module, ["--tokenizer_name", tokenizer, "--dialect_tokenizer_dir", run / "dialect_tokenizer",
                                      "--train_corpus", run / "region_data/corpus/dialect_train_corpus.txt",
                                      "--validation_corpus", run / "region_data/corpus/dialect_validation_corpus.txt",
                                      "--dataset_cache_dir", dataset_cache,
                                      "--output_dir", output, *common])
        patch_short_training(module)
        module.pretrain(args)
    elif stage == "classify":
        output = region_out / "classifiers" / tokenizer / f"seed_{SEED}"
        args = parse_original(module, ["--mlm_model_dir", region_out / "mlm" / tokenizer / "final_model",
                                      "--train_tsv", run / "region_data/region_classification/dialect_region_train.tsv",
                                      "--validation_tsv", run / "region_data/region_classification/dialect_region_validation.tsv",
                                      "--test_tsv", run / "region_data/region_classification/dialect_region_test.tsv",
                                      "--dataset_cache_dir", dataset_cache,
                                      "--tokenized_cache_dir", run / "cache" / f"classification_tokenized_{tokenizer}",
                                      "--seed", str(SEED), "--output_dir", output, *common])
        patch_short_training(module)
        module.fine_tune(args)
    elif stage == "changed":
        def must_not_rebuild(*_args, **_kwargs):
            raise RuntimeError("Smoke changed-subset analysis attempted to reopen source JSON instead of using the canonical cache")
        module.rebuild_changed_flags = must_not_rebuild
        args = parse_original(module, ["--split_manifest", run / "region_data/corpus_split_manifest_80_10_10.csv",
                                      "--test_tsv", run / "region_data/region_classification/dialect_region_test.tsv",
                                      "--outputs_root", region_out, "--result_dir", run / "region_results",
                                      "--seeds", str(SEED)])
        module.analyze(args)
        result = read_json(run / "region_results/changed_subset_results.json")["composition"]
        if result["changed_rows"] != 5 or result["changed_fraction_raw"] != 0.0:
            raise RuntimeError("Changed-subset analysis did not use canonical changed flags")
    elif stage == "region_summary":
        args = parse_original(module, ["--output_root", region_out, "--result_dir", run / "region_results",
                                      "--logs_dir", run / "logs", "--seeds", str(SEED)])
        module.summarize(args)
        summary = read_json(run / "region_results/final_results.json")
        if set(summary["tokenizers"]) != set(TOKENIZERS) or summary["seeds"] != [SEED]:
            raise RuntimeError("Region summary did not include all four smoke conditions")
        result = {"summary_exists": True, "summary_conditions_checked": list(TOKENIZERS)}
    elif stage == "decoder":
        output = translation_out / "shared_standard_decoder"
        args = parse_original(module, ["--train_corpus", run / "translation_data/corpus/standard_train_corpus.txt",
                                      "--validation_corpus", run / "translation_data/corpus/standard_validation_corpus.txt",
                                      "--dataset_cache_dir", dataset_cache, "--output_dir", output, *common])
        patch_short_training(module)
        module.pretrain(args)
    elif stage == "translate":
        output = translation_out / "translation" / tokenizer / f"seed_{SEED}"
        args = parse_original(module, ["--tokenizer_name", tokenizer, "--seed", str(SEED),
                                      "--source_mlm_model_dir", region_out / "mlm" / tokenizer / "final_model",
                                      "--shared_decoder_model_dir", translation_out / "shared_standard_decoder/final_model",
                                      "--train_tsv", run / "translation_data/translation/translation_train.tsv",
                                      "--validation_tsv", run / "translation_data/translation/translation_validation.tsv",
                                      "--test_tsv", run / "translation_data/translation/translation_test.tsv",
                                      "--dataset_cache_dir", dataset_cache,
                                      "--tokenized_cache_dir", run / "cache" / f"translation_tokenized_{tokenizer}",
                                      "--generation_max_length", "16", "--validation_generation_samples", "10",
                                      "--output_dir", output, *common])
        patch_short_training(module)
        module.train(args)
    elif stage == "translation_summary":
        # The original summary's only seed selector is a module constant.
        # Override it in this isolated smoke process; the source file is unchanged.
        module.SEEDS = [SEED]
        args = parse_original(module, ["--outputs_root", translation_out, "--logs_dir", run / "logs",
                                      "--result_dir", run / "translation_results"])
        module.summarize(args)
        summary = read_json(run / "translation_results/final_results.json")
        if set(summary["test"]) != set(TOKENIZERS):
            raise RuntimeError("Translation summary did not include all four smoke conditions")
        result = {"summary_exists": True, "summary_conditions_checked": list(TOKENIZERS)}
    else:
        raise ValueError(stage)
    if output is not None:
        result = verify_training_and_reload(stage, output)
        if stage == "classify":
            metrics = read_json(output / "test_metrics.json")
            if not all(math.isfinite(float(metrics[key])) for key in ("accuracy", "macro_f1")):
                raise RuntimeError("Classification metrics are nonfinite")
            result["classification_metrics_checked"] = True
        if stage == "translate":
            metrics = read_json(output / "test_generation_report.json")
            if metrics["overall"]["count"] != 10 or metrics["changed_only"]["count"] != 5:
                raise RuntimeError("Translation evaluation row/subset counts are wrong")
            if not all(math.isfinite(float(metrics["overall"][key])) for key in ("chrf_plus_plus", "sacrebleu", "cer", "exact_match")):
                raise RuntimeError("Generation metrics are nonfinite")
            result["generation_metrics_checked"] = True
    save_json(run / "stage_reports" / f"{stage}_{tokenizer or 'common'}.json", {"completed": True, **result})


def run_command(command, log, environment):
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code != 0:
        raise RuntimeError(f"Stage exited with code {code}; see {log}")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="CPU functional run with FP32 and non-fused AdamW; does not certify CUDA.")
    parser.add_argument("--dry-run", action="store_true", help="Check source hashes/syntax and show stage plan; does NOT train or pass the smoke test.")
    parser.add_argument("--stage", choices=("region", "translation", "all"), default="region",
                        help="Run region stages in dialect1, then translation stages in dialect_translate.")
    parser.add_argument("--region-run", type=Path,
                        help="Completed isolated region smoke directory, required for --stage translation.")
    parser.add_argument("--_stage", choices=list(SCRIPTS), help=argparse.SUPPRESS)
    parser.add_argument("--_tokenizer", choices=TOKENIZERS, help=argparse.SUPPRESS)
    parser.add_argument("--_run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args._stage:
        if args._run_dir is None:
            raise ValueError("Internal stage requires isolated --_run-dir")
        execute_child(args._stage, args._tokenizer, args._run_dir.resolve(), args.cpu)
        return 0
    hashes = original_hashes()
    stages = plan(args.stage)
    if args.dry_run:
        for number, (stage, tokenizer) in enumerate(stages, 1):
            print(f"{number:02d}. {stage}: {tokenizer or 'common'}")
        print(f"[DRY RUN ONLY] {len(hashes)} original hashes/syntax verified. No training or smoke PASS claimed.")
        return 0
    source_run = None
    if args.stage == "translation":
        if args.region_run is None:
            parser.error("--stage translation requires --region-run from a completed region smoke")
        source_run = args.region_run.resolve()
        if not source_run.is_relative_to(SMOKE_ROOT.resolve()):
            parser.error("--region-run must be inside server/smoke_runs")
        previous = read_json(source_run / "smoke_report.json")
        if previous.get("completed") is not True or previous.get("selected_stage") not in ("region", "all"):
            parser.error("The selected region smoke has not completed successfully")
        if previous.get("original_source_sha256") != hashes:
            parser.error("Region smoke was created from different original experiment code")
    elif args.region_run is not None:
        parser.error("--region-run is only used with --stage translation")
    # A short run/cache prefix also supports Windows hosts without long paths.
    run = SMOKE_ROOT / (datetime.now(timezone.utc).strftime("%y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4])
    run.mkdir(parents=True, exist_ok=False)
    report = {"completed": False, "status": "running", "cpu_only": args.cpu,
              "selected_stage": args.stage, "region_run": str(source_run) if source_run else None,
              "production_artifacts_used": False, "run_dir": str(run), "stages": [],
              "original_source_sha256": hashes, "started_utc": datetime.now(timezone.utc).isoformat()}
    save_json(run / "smoke_report.json", report)
    environment = os.environ.copy()
    environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", TOKENIZERS_PARALLELISM="false",
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    current = "environment"
    try:
        command = [sys.executable, str(ROOT / "environment_check.py"), "--report", str(run / "environment_report.json"),
                   "--stage", args.stage]
        if args.cpu:
            command.append("--cpu")
        run_command(command, run / "environment.log", environment)
        # Datasets embeds the absolute cache path in Windows lock filenames.
        # A unique short temporary cache avoids MAX_PATH without changing the OS.
        dataset_cache = Path(tempfile.mkdtemp(prefix="kdsm_")) if os.name == "nt" else run / "c"
        report["dataset_cache"] = str(dataset_cache)
        save_json(run / "smoke_report.json", report)
        make_fixture(run)
        for number, (stage, tokenizer) in enumerate(stages, 1):
            current = f"{stage}_{tokenizer or 'common'}"
            print(f"[SMOKE {number}/{len(stages)}] {current}", flush=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--_stage", stage, "--_run-dir", str(run)]
            if tokenizer:
                command += ["--_tokenizer", tokenizer]
            if args.cpu:
                command.append("--cpu")
            started = time.perf_counter()
            run_command(command, run / "logs" / f"{number:02d}_{current}.log", environment)
            result = read_json(run / "stage_reports" / f"{current}.json")
            report["stages"].append({"stage": current, "seconds": time.perf_counter() - started, **result})
            save_json(run / "smoke_report.json", report)
        backbone_hashes = {
            read_json((source_run or run) / "region_outputs/mlm" / tokenizer / "mlm_pretraining_metadata.json")["shared_backbone_sha256"]
            for tokenizer in TOKENIZERS
        }
        if len(backbone_hashes) != 1:
            raise RuntimeError("The four MLM conditions did not share identical initial backbone weights")
        if original_hashes() != hashes:
            raise RuntimeError("Original experiment files changed during smoke execution")
        report.update(completed=True, status="passed", finished_utc=datetime.now(timezone.utc).isoformat(),
                      original_shapes_verified=True, shared_backbone_initialization_verified=True,
                      cuda_validated=not args.cpu, external_gpu_monitor_validated=False,
                      production_32000_tokenizer_trained=False)
        save_json(run / "smoke_report.json", report)
        print(f"[SMOKE PASS] {args.stage}, {'CPU functional' if args.cpu else 'CUDA'} execution: {run / 'smoke_report.json'}")
        print("[SCOPE] Synthetic 128-token KRD vocabulary, two steps, one seed; these are not production model/results.")
        return 0
    except BaseException as exc:
        report.update(completed=False, status="failed", failed_stage=current, error=str(exc), traceback=traceback.format_exc())
        save_json(run / "smoke_report.json", report)
        print(f"[SMOKE FAIL] {current}: {exc}\nReport: {run / 'smoke_report.json'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
