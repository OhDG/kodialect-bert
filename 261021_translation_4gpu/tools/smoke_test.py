#!/usr/bin/env python3
"""Offline training smoke; a CPU-only pass never certifies CUDA readiness.

Default: parallel, independent one-GPU workers and four synthetic tokenizer
variants. Each worker trains CLM for two optimizer steps, reloads its checkpoint
and resumes to step three, then does the same for its assigned seq2seq variants.
--require-inputs checks the transferred tokenizers using the production loader.
--production-shapes additionally loads the real MLM encoders and uses the
production 6-layer/384-wide decoder. This is still a small-batch functional test,
not a claim that the production batch size fits in GPU memory. Add
--production-batches to also probe full 128-token training/evaluation batches
and four-beam generation before starting a long experiment.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("dialect", "klue", "kobert", "mbert")
EXPECTED = {
    "torch": "2.0.1+cu118", "transformers": "4.40.1", "datasets": "2.19.1",
    "tokenizers": "0.19.1", "numpy": "1.26.4", "pyarrow": "15.0.2",
    "accelerate": "0.30.1", "sentencepiece": "0.2.0", "sacrebleu": "2.6.0",
    "rapidfuzz": "3.14.5",
}
TEXTS = ["오늘 날씨 정말 좋다", "학교에 가요", "밥 먹었어요", "안녕하세요"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def environment_report(cpu=False):
    import torch
    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    mismatches = {name: {"expected": expected, "actual": versions[name]}
                  for name, expected in EXPECTED.items() if versions[name] != expected}
    report = {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "versions": versions, "version_mismatches": mismatches,
        "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_only": cpu, "gpu_readiness_certified": False,
    }
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total",
                                "--format=csv,noheader"], capture_output=True, text=True, check=False)
        report["nvidia_smi"] = query.stdout.strip() or query.stderr.strip()
    except OSError as exc:
        report["nvidia_smi"] = str(exc)
    if not cpu:
        if sys.version_info[:2] != (3, 10) or mismatches:
            raise RuntimeError(f"Use the pinned Python 3.10 environment: {mismatches}; Python={sys.version}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. A CPU run cannot certify GPU readiness.")
        if torch.version.cuda != "11.8":
            raise RuntimeError(f"Expected the cu118 wheel, got CUDA {torch.version.cuda}")
    return report


def make_fixture_tokenizer(directory, variant):
    from transformers import BertTokenizerFast
    directory.mkdir(parents=True, exist_ok=True)
    # Deliberately different source IDs and vocabulary sizes exercise dual-tokenizer use.
    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    words = list(dict.fromkeys(" ".join(TEXTS).split()))
    shift = VARIANTS.index(variant)
    words = words[shift:] + words[:shift]
    tokens = special + words + [f"unused_{variant}_{i}" for i in range(16 + 8 * shift)]
    vocab = directory / "vocab.txt"
    vocab.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False, tokenize_chinese_chars=False)
    tokenizer.save_pretrained(directory)
    return tokenizer


def tokenizer_roundtrip(tokenizer, directory):
    from transformers import AutoTokenizer
    from experiment_common import save_tokenizer_compatible, tokenizer_fingerprint
    save_tokenizer_compatible(tokenizer, directory)
    restored = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=True)
    assert tokenizer(TEXTS)["input_ids"] == restored(TEXTS)["input_ids"], "Tokenizer round-trip changed IDs"
    assert tokenizer_fingerprint(tokenizer) == tokenizer_fingerprint(restored), "Vocabulary round-trip changed"
    return {"class": type(tokenizer).__name__, "size": len(tokenizer),
            "fingerprint": tokenizer_fingerprint(tokenizer), "pad_id": tokenizer.pad_token_id,
            "cls_id": tokenizer.cls_token_id, "sep_id": tokenizer.sep_token_id}


def model_config(tokenizer, production=False, decoder=False):
    from transformers import BertConfig
    return BertConfig(vocab_size=len(tokenizer), hidden_size=384 if production else 32,
                      num_hidden_layers=6 if production else 2, num_attention_heads=6 if production else 4,
                      intermediate_size=1536 if production else 64, max_position_embeddings=512,
                      hidden_dropout_prob=0.1 if production else 0.0,
                      attention_probs_dropout_prob=0.1 if production else 0.0,
                      pad_token_id=tokenizer.pad_token_id, is_decoder=decoder, add_cross_attention=decoder)


def training_args(output, cpu, seq2seq=False, max_steps=2):
    from transformers import TrainingArguments, Seq2SeqTrainingArguments
    cls = Seq2SeqTrainingArguments if seq2seq else TrainingArguments
    kwargs = dict(output_dir=str(output), max_steps=max_steps, per_device_train_batch_size=2,
                  per_device_eval_batch_size=2, gradient_accumulation_steps=1,
                  learning_rate=1e-4, lr_scheduler_type="constant",
                  save_strategy="steps", save_steps=2, save_total_limit=2,
                  logging_steps=1, evaluation_strategy="no", report_to="none", disable_tqdm=True,
                  fp16=not cpu, tf32=not cpu, optim="adamw_torch" if cpu else "adamw_torch_fused",
                  dataloader_num_workers=0, seed=13, data_seed=13, save_safetensors=True,
                  remove_unused_columns=False, no_cuda=cpu)
    if seq2seq:
        kwargs.update(predict_with_generate=True, generation_max_length=12, generation_num_beams=2)
    return cls(**kwargs)


def production_memory_probe(model, source, target, seq2seq=False):
    """Exercise full configured tensor shapes, including persistent Adam state."""
    import torch
    from step3_train_dialect_to_standard import DualTokenizerSeq2SeqCollator

    def full_ids(tokenizer, is_source=False):
        body = tokenizer(" ".join(TEXTS), add_special_tokens=False)["input_ids"]
        assert body, "Cannot construct a nonempty memory-probe fixture"
        # Source/CLM has CLS + 126 tokens + SEP; seq2seq labels have 127 + SEP.
        count = 126 if is_source else 127
        repeated = (body * ((count + len(body) - 1) // len(body)))[:count]
        return ([tokenizer.cls_token_id] if is_source else []) + repeated + [tokenizer.sep_token_id]

    source_ids = full_ids(source, is_source=True)
    target_ids = full_ids(target)
    model.to("cuda")
    if seq2seq:
        feature = {"input_ids": source_ids, "attention_mask": [1] * 128, "labels": target_ids}
        collator = DualTokenizerSeq2SeqCollator(source, target, model)
        train_batch = {key: value.cuda() for key, value in collator([feature] * 64).items()}
    else:
        ids = torch.tensor(full_ids(target, is_source=True), device="cuda").repeat(64, 1)
        train_batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
    assert train_batch["input_ids"].shape == (64, 128)
    assert train_batch["labels"].shape == (64, 128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    scaler = torch.cuda.amp.GradScaler()
    optimizer.zero_grad(set_to_none=True)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    losses = []
    for _ in range(4):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            result = model(**train_batch)
            loss = result.loss
        assert torch.isfinite(loss).item(), "Non-finite loss at production batch shape"
        losses.append(float(loss.detach()))
        scaler.scale(loss / 4).backward()
        del result, loss
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    steps = [int(state["step"]) for state in optimizer.state.values() if "step" in state]
    assert steps and max(steps) == 1, "Production-shape optimizer step was skipped (FP16 overflow)"
    report = {"train_batch_size": 64, "gradient_accumulation_steps": 4, "sequence_length": 128,
              "optimizer_steps": 1, "fused_optimizer": True, "precision": "fp16",
              "train_losses": losses, "train_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "train_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    optimizer.zero_grad(set_to_none=True)
    del train_batch
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    if seq2seq:
        evaluation = {key: value.cuda() for key, value in collator([feature] * 32).items()}
    else:
        ids = torch.tensor(full_ids(target, is_source=True), device="cuda").repeat(128, 1)
        evaluation = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        output = model(**evaluation)
        assert torch.isfinite(output.loss).item(), "Non-finite production batch eval loss"
        report["eval_loss"] = float(output.loss)
    del output
    report["eval_batch_size"] = 32 if seq2seq else 128
    report["eval_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    if seq2seq:
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            generated = model.generate(input_ids=evaluation["input_ids"], attention_mask=evaluation["attention_mask"],
                                       num_beams=4, do_sample=False, min_length=128, max_length=128)
        assert generated.shape == (32, 128), "Generation did not exercise the full requested shape"
        report.update(generation_batch_size=32, num_beams=4, generation_length=128,
                      generation_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      generation_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        del generated
    report["passed"] = True
    del evaluation, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return report


def train_resume(model, dataset, collator_factory, tokenizer, output, cpu, seq2seq=False):
    import torch
    from transformers import Trainer, Seq2SeqTrainer
    from experiment_common import enable_trusted_local_resume
    cls = Seq2SeqTrainer if seq2seq else Trainer
    trainer = cls(model=model, args=training_args(output, cpu, seq2seq), train_dataset=dataset,
                  eval_dataset=dataset.select(range(2)), data_collator=collator_factory(model), tokenizer=tokenizer)
    result = trainer.train()
    assert trainer.state.global_step == 2 and math.isfinite(result.training_loss)
    if not cpu:
        assert next(model.parameters()).device.type == "cuda"
    checkpoint = output / "checkpoint-2"
    for required in ("trainer_state.json", "optimizer.pt", "scheduler.pt"):
        assert (checkpoint / required).is_file(), f"Missing resume state: {required}"
    saved = output / "saved_model"
    trainer.save_model(saved)
    model_cls = type(model)
    del trainer, model
    gc.collect()
    if not cpu:
        torch.cuda.empty_cache()
    restored = model_cls.from_pretrained(saved, local_files_only=True)
    before_resume = next(restored.parameters()).detach().cpu().clone()
    enable_trusted_local_resume(str(checkpoint), str(output))
    resumed = cls(model=restored, args=training_args(output, cpu, seq2seq, max_steps=3), train_dataset=dataset,
                  eval_dataset=dataset.select(range(2)), data_collator=collator_factory(restored), tokenizer=tokenizer)
    resumed_result = resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 3 and math.isfinite(resumed_result.training_loss)
    assert not torch.equal(before_resume, next(restored.parameters()).detach().cpu()), "Resume did not update model weights"
    optimizer = getattr(resumed.optimizer, "optimizer", resumed.optimizer)
    optimizer_step = max(int(state["step"]) for state in optimizer.state.values() if "step" in state)
    assert optimizer_step == 3, "Optimizer state did not resume from step two"
    if not cpu:
        assert optimizer.defaults.get("fused") is True, "Fused AdamW was not enabled"
        assert resumed.accelerator.mixed_precision == "fp16", "FP16 was not enabled"
    evaluated = resumed.evaluate()
    assert math.isfinite(evaluated["eval_loss"]), "Non-finite eval loss"
    return resumed, {"initial_steps": 2, "resumed_step": 3, "optimizer_step": optimizer_step,
                     "fused_optimizer": bool(optimizer.defaults.get("fused")),
                     "precision": "fp32" if cpu else "fp16", "initial_loss": result.training_loss,
                     "resumed_loss": resumed_result.training_loss, "evaluation": evaluated}


def worker(args):
    # Set offline controls before importing HF or torch. CUDA visibility is set by the parent.
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", WANDB_DISABLED="true")
    sys.path.insert(0, str(ROOT / "engine"))
    import torch
    from datasets import Dataset, load_from_disk
    from transformers import AutoModel, BertModel, BertLMHeadModel, EncoderDecoderModel, DataCollatorForLanguageModeling
    from experiment_common import configure_reproducibility, generation_metrics, load_local_tokenizer
    from step3_train_dialect_to_standard import DualTokenizerSeq2SeqCollator

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    report = environment_report(args.cpu)
    report.update(status="running", variants=args.variants, production_shapes=args.production_shapes,
                  actual_input_tokenizers=args.require_inputs, production_batch_size_tested=False)
    configure_reproducibility(13)
    torch.set_num_threads(2)
    if not args.cpu:
        assert torch.cuda.device_count() == 1, "Each smoke worker must see exactly one GPU"
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        report["gpu"] = {"name": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0),
                         "total_bytes": torch.cuda.get_device_properties(0).total_memory}
        torch.cuda.reset_peak_memory_stats()

    def get_tokenizer(variant):
        if args.require_inputs:
            path = args.inputs_root / variant / "final_model"
            if not path.is_dir():
                raise FileNotFoundError(f"Transferred model folder missing: {path}")
            return load_local_tokenizer(path, 32)
        return make_fixture_tokenizer(out / "fixture_tokenizers" / variant, variant)

    target = get_tokenizer("klue")
    assert target.pad_token_id is not None and target.cls_token_id is not None and target.sep_token_id is not None
    report["target_tokenizer"] = tokenizer_roundtrip(target, out / "roundtrip_target")
    decoder = BertLMHeadModel(model_config(target, args.production_shapes, decoder=True))
    if args.production_batches:
        report["stage"] = "clm_production_memory_probe"
        write_json(out / "report.json", report)
        print("[MEMORY PROBE] CLM: train64 x 128 x accumulation4; eval128 x 128", flush=True)
        report["production_memory_probes"] = {
            "clm": production_memory_probe(decoder, target, target), "seq2seq": {}}
        write_json(out / "report.json", report)
    clm_rows = target(TEXTS * 4, truncation=True, max_length=24, return_special_tokens_mask=True)
    clm_data = Dataset.from_dict(clm_rows)
    clm_data.save_to_disk(str(out / "clm_dataset"))
    clm_data = load_from_disk(str(out / "clm_dataset"))
    clm, report["clm"] = train_resume(decoder, clm_data,
        lambda model: DataCollatorForLanguageModeling(target, mlm=False, pad_to_multiple_of=8),
        target, out / "clm", args.cpu)
    decoder_path = out / "clm" / "resumed_decoder"
    clm.save_model(decoder_path)
    report["seq2seq"] = {}
    del clm, decoder
    gc.collect()
    if not args.cpu:
        torch.cuda.empty_cache()

    for variant in args.variants:
        source = get_tokenizer(variant)
        source_info = tokenizer_roundtrip(source, out / f"roundtrip_{variant}")
        encoder = (AutoModel.from_pretrained(args.inputs_root / variant / "final_model", local_files_only=True)
                   if args.production_shapes else BertModel(model_config(source)))
        assert max(source.get_vocab().values()) < encoder.get_input_embeddings().num_embeddings
        decoder = BertLMHeadModel.from_pretrained(decoder_path, local_files_only=True)
        model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
        model.config.decoder_start_token_id = target.cls_token_id
        model.config.eos_token_id = target.sep_token_id
        model.config.pad_token_id = target.pad_token_id
        model.config.vocab_size = len(target)
        for config in (model.generation_config,):
            config.decoder_start_token_id = target.cls_token_id
            config.eos_token_id = target.sep_token_id
            config.pad_token_id = target.pad_token_id
        if args.production_batches:
            report["stage"] = f"{variant}_production_memory_probe"
            write_json(out / "report.json", report)
            print(f"[MEMORY PROBE] {variant}: train64 x 128 x accumulation4; generation32 x 128 beam4", flush=True)
            report["production_memory_probes"]["seq2seq"][variant] = production_memory_probe(
                model, source, target, seq2seq=True)
            write_json(out / "report.json", report)
        source_rows = source(TEXTS * 4, truncation=True, max_length=24)
        labels = target(TEXTS * 4, add_special_tokens=False, truncation=True, max_length=23)["input_ids"]
        features = [{"input_ids": ids, "attention_mask": mask, "labels": label + [target.sep_token_id]}
                    for ids, mask, label in zip(source_rows["input_ids"], source_rows["attention_mask"], labels)]
        collate = lambda current: DualTokenizerSeq2SeqCollator(source, target, current)
        sample = collate(model)(features[:2])
        assert sample["input_ids"].shape[1] % 8 == 0 and sample["labels"].shape[1] % 8 == 0
        assert sample["labels"].eq(-100).any(), "Label padding must use -100"
        assert sample["decoder_input_ids"][:, 0].eq(target.cls_token_id).all()
        assert torch.equal(sample["decoder_attention_mask"], sample["decoder_input_ids"].ne(target.pad_token_id))
        dataset = Dataset.from_list(features)
        trainer, result = train_resume(model, dataset, collate, target, out / variant, args.cpu, seq2seq=True)
        batch = source(TEXTS[:2], return_tensors="pt", padding=True)
        batch = {key: value.to(trainer.args.device) for key, value in batch.items() if key in ("input_ids", "attention_mask")}
        trainer.model.eval()
        with torch.no_grad():
            generated = trainer.model.generate(**batch, max_new_tokens=8, num_beams=2)
        predicted = target.batch_decode(generated.cpu(), skip_special_tokens=True)
        metrics = generation_metrics(predicted, TEXTS[:2])
        assert metrics["count"] == 2 and all(math.isfinite(float(value)) for value in metrics.values())
        identity = generation_metrics(TEXTS, TEXTS)
        assert identity["exact_match"] == 1.0 and identity["cer"] == 0.0
        result.update(source_tokenizer=source_info, generated=predicted, metrics=metrics,
                      encoder_config={key: getattr(encoder.config, key, None) for key in
                                      ("vocab_size", "hidden_size", "num_hidden_layers")})
        report["seq2seq"][variant] = result
        write_json(out / "report.json", report)
        del trainer, model, encoder, decoder
        gc.collect()
        if not args.cpu:
            torch.cuda.empty_cache()
    if not args.cpu:
        torch.cuda.synchronize()
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["gpu_readiness_certified"] = True
    report["status"] = "cpu_only_passed" if args.cpu else "gpu_passed"
    report["stage"] = "complete"
    report["production_batch_size_tested"] = args.production_batches
    write_json(out / "report.json", report)
    print(json.dumps({"status": report["status"], "output": str(out)}, ensure_ascii=False), flush=True)


def orchestrate(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Use a fresh smoke output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    devices = ["cpu"] if args.cpu else args.gpus
    jobs = []
    assignments = {device: [] for device in devices}
    for index, variant in enumerate(args.variants):
        assignments[devices[index % len(devices)]].append(variant)
    # Even an extra requested GPU gets a real CLM and seq2seq smoke.
    for device in devices:
        variants = assignments[device] or [args.variants[0]]
        directory = args.output_dir / ("cpu" if args.cpu else f"gpu_{device}")
        directory.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--output-dir", str(directory),
                   "--inputs-root", str(args.inputs_root), "--variants", *variants]
        for name in ("cpu", "require_inputs", "production_shapes", "production_batches"):
            if getattr(args, name):
                command.append("--" + name.replace("_", "-"))
        env = os.environ.copy()
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(key, None)
        env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="" if args.cpu else device,
                   PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        log = (directory / "smoke.log").open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        except BaseException:
            log.close()
            for _, running, handle, _ in jobs:
                running.terminate()
                running.wait()
                handle.close()
            raise
        jobs.append((device, process, log, directory))
        print(f"Started {device}: {', '.join(variants)}; log={directory / 'smoke.log'}", flush=True)
    results = []
    try:
        for device, process, log, directory in jobs:
            code = process.wait()
            log.close()
            report_path = directory / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
            expected_status = "cpu_only_passed" if args.cpu else "gpu_passed"
            passed = code == 0 and report.get("status") == expected_status
            results.append({"device": device, "returncode": code, "passed": passed,
                            "report": str(report_path), "log": str(directory / "smoke.log")})
            print(f"{device}: {'PASS' if passed else 'FAIL'}", flush=True)
    finally:
        for _, process, log, _ in jobs:
            if process.poll() is None:
                process.terminate()
                process.wait()
            log.close()
    passed = all(item["passed"] for item in results)
    summary = {"status": ("cpu_only_passed" if args.cpu else "gpu_passed") if passed else "failed",
               "gpu_readiness_certified": passed and not args.cpu,
               "production_batch_size_tested": passed and args.production_batches, "workers": results}
    write_json(args.output_dir / "summary.json", summary)
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"], help="physical GPU indices")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "smoke" / time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--inputs-root", type=Path, default=ROOT / "models" / "mlm")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--require-inputs", action="store_true", help="use all assigned real source tokenizers plus KLUE target")
    parser.add_argument("--production-shapes", action="store_true", help="load real MLM weights and use 6x384 decoder; implies --require-inputs")
    parser.add_argument("--production-batches", action="store_true", help="GPU-only 128-token memory probes: train64 x accumulation4, CLM eval128, seq2seq eval/generate32 beam4; implies --production-shapes")
    parser.add_argument("--cpu", action="store_true", help="explicit local functional test only; permits version drift, does not certify GPUs")
    parser.add_argument("--check-environment", action="store_true", help="check pinned versions and CUDA visibility without training")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.inputs_root = args.inputs_root.resolve()
    args.production_shapes = args.production_shapes or args.production_batches
    args.require_inputs = args.require_inputs or args.production_shapes
    if args.cpu and args.production_batches:
        parser.error("--production-batches requires real GPUs; --cpu cannot certify production memory usage")
    if len(set(args.gpus)) != len(args.gpus) or any(not re.fullmatch(r"0|[1-9][0-9]*", gpu) for gpu in args.gpus):
        parser.error("--gpus must contain unique nonnegative integer indices")
    try:
        if args.check_environment:
            report = environment_report(args.cpu)
            report["status"] = "environment_checked"
            write_json(args.output_dir / "environment.json", report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        if args.worker:
            worker(args)
            return 0
        return orchestrate(args)
    except Exception as exc:
        if args.worker or args.check_environment:
            report_path = args.output_dir / "report.json"
            previous = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
            previous.update(status="failed", gpu_readiness_certified=False, production_batch_size_tested=False,
                            error=str(exc), traceback=traceback.format_exc())
            write_json(report_path, previous)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
