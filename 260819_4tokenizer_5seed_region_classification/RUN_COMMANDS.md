# 260819 four-tokenizer, five-seed region classification

## Why this directory exists

`260818_4tokenizer_5seed_region_classification` is cancelled. It attempted an
aggressive batch-size redesign (train batch 64→256, eval batch 256/2048→2,048/512)
to exploit this A6000's full 48 GB, and crashed twice:

1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — an allocator option this
   server's PyTorch build does not recognize, causing a hard `RuntimeError` at CUDA
   init.
2. An MLM evaluation batch that ignored vocabulary size: a `[2,048, 128, 32000]`
   cross-entropy logits tensor needs 31.25 GiB by itself, causing a real
   `CUDA out of memory` at the first epoch-end evaluation.

This directory abandons that redesign entirely. It runs `260807`'s **original,
unmodified batch profile and hyperparameters**, upgraded to process-only GPU
measurement: it records only this experiment's own process (via `pynvml`,
ported from the `260814_..._local` / `260818` work) instead of whole-device
`nvidia-smi` snapshots that can include other users' jobs on a shared GPU.

**This directory trains everything from scratch by default — dialect tokenizer,
all four MLM encoders, and all four×five classifier fine-tuning runs.** An
earlier version of this directory defaulted to reusing `260807`'s already-
completed encoders instead (skipping MLM entirely), on the reasoning that
`260811` (translation) already fine-tunes against those same encoders. That
reasoning about the *encoder weights* still holds — reuse wouldn't have
corrupted anything trained on top of them. But `260807`'s own
`results/final_results.md` turned out to show clear whole-device GPU
measurement contamination:

- dialect's MLM peak VRAM: 47.38 / 47.40 GB — ~100% of the card's total
  capacity, for a 23M-parameter model.
- dialect's classifier peak VRAM: 24.11 GB vs KLUE's *identical-architecture*
  12.95 GB.
- dialect's classifier time: 2.45 ± 0.93 h — a 38% relative standard
  deviation across 5 seeds that should be near-deterministic (KLUE's was
  1.69 ± 0.00 h).
- mBERT's classifier peak VRAM (11.24 GB) was the *lowest* of the four despite
  having the largest vocabulary and thus the largest embedding table.

Cross-checked against `260814_..._local`'s own process-only measurements
(2.4–3.1 GB classifier peak allocated across all four tokenizers, no anomalies),
the gap is too large to be allocator caching — something else was sharing the
GPU during at least parts of that run. Given that, we're training everything
here from scratch instead of reusing, so every number in this run — MLM and
classifier both — comes from this run's own clean, process-only measurement,
not inherited from a run with contamination in its telemetry.

`260807`'s **Accuracy/Macro F1 tables** (not the GPU efficiency table) are still
usable as-is if needed for a quick cross-check — GPU contention slows wall time,
it doesn't corrupt the trained weights or the resulting metrics. But this run is
the one to cite for both the final performance numbers and any efficiency claim.

Reuse is still available as an opt-in if ever needed again:

```bash
python run_full_experiment.py --reuse_mlm_from "../260807_4tokenizer_5seed_region_classification"
```

## Experiment design (identical to 260807)

- File-level split: Train 80%, Validation 10%, Test 10%
- Tokenizers: dialect WordPiece, KLUE-BERT, KoBERT, multilingual BERT
- Pretrained weights from the baseline models are not used.
- Each tokenizer trains the same 6-layer Small BERT from scratch with MLM for 3 epochs.
- MLM is trained once with seed 42 for each tokenizer.
- Region classification is repeated with seeds 13, 21, 42, 87, and 100.
- The best epoch is selected by Validation Macro F1.
- The selected checkpoint is evaluated once on the independent Test split.
- Final values are reported as the five-seed mean and standard deviation.

## Server paths

Run from:

```bash
cd /data/ohdg/kodialect-bert/260819_4tokenizer_5seed_region_classification
```

The default source manifest is:

```text
../260630_test_1/corpus_split_manifest.csv
```

Its existing JSON paths are reused to generate the new split.

## One-time dependency and smoke check

```bash
python -m pip install -r requirements_260819.txt
python run_full_experiment.py --smoke
```

The smoke command still prepares the 80/10/10 data when it does not already exist. Subsequent runs reuse it.

## Full experiment

```bash
python run_full_experiment.py
```

This runs the full pipeline: data prep, dialect tokenizer training, MLM
pretraining for all four tokenizers, then classifier fine-tuning (4 tokenizers
× 5 seeds = 20 Trainer runs) — all with process-only GPU measurement. The
runner is resumable: re-entering the same command skips completed runs and
resumes the latest complete Trainer checkpoint for an interrupted stage. Do not
add `--overwrite` to a normal resume command — it discards completed stages.

For a persistent visible terminal session:

```bash
tmux new -s dialect_260819
python run_full_experiment.py
```

Detach with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t dialect_260819
```

## A6000 profile (unchanged from 260807 — no redesign)

| Stage | Tokenizer | Micro batch | Accumulation | Effective batch | Eval batch |
| --- | --- | ---: | ---: | ---: | ---: |
| MLM | dialect/KLUE/KoBERT | 256 | 1 | 256 | 512 |
| MLM | mBERT | 128 | 2 | 256 | 256 |
| Classification | all | 256 | 1 | 256 | 2,048 |

Common settings: maximum length 128, FP16, TF32, fused AdamW, 8 DataLoader workers, and 16 tokenizer workers.

These eval batch values are not arbitrary: MLM cross-entropy materializes a
`[eval_batch, seq_len, vocab_size]` logits tensor, so cost scales with
`eval_batch × vocab_size`. At eval=512/32,000-vocab that's ~7.8 GiB; at
eval=256/119,547-vocab (mBERT) that's ~14.6 GiB — both comfortably inside 48 GB.
The 260818 attempt broke this exact relationship by scaling eval batch as if it
were as cheap as a bigger GPU makes train batch; it isn't, because vocab size
(not GPU class) sets the ceiling here. Do not raise these two eval batch values
without recomputing the logits tensor size first.

## Outputs

```text
data/                         New 80/10/10 corpus and TSV files
dialect_bert_tokenizer/       Train-only dialect tokenizer
outputs/mlm/                  Four MLM models
outputs/classifiers/          Four tokenizers x five seeds
cache/                        Reusable tokenized datasets
logs/                         Console logs and one-second GPU samples
results/final_results.md      Final paper-ready summary
results/final_results.json    Full aggregated result
results/overall_by_seed.csv   Per-seed Test metrics
results/overall_summary.csv   Mean and standard deviation
results/efficiency_summary.csv Process-only GPU/runtime summary
```

Every MLM and classifier stage records process-only peak allocated/reserved
VRAM and process SM utilization in `logs/*_gpu_summary.json` and in
`outputs/.../*_metadata.json` (`process_gpu_memory`, from
`torch.cuda.max_memory_allocated`/`max_memory_reserved` inside the training
process itself — cannot include another user's job even on a shared card).
Verify this held in practice by checking `nvml_process_vram_supported: true` in
the `*_gpu_summary.json` files after the run.

## Important interpretation

The primary controlled comparison is dialect versus KLUE because both use a 32,000-token vocabulary.
KoBERT and mBERT are native-vocabulary reference tokenizers; their native vocabulary sizes change the embedding and MLM
output parameter counts. All four receive equal MLM epochs, classification seeds, data, and effective batch sizes.

If another job runs on this GPU concurrently during this run, process-only
measurement means it cannot leak into these numbers the way it apparently did
in `260807`'s original run — but wall-clock *time* can still be slowed by
contention even though memory/utilization readings stay clean. If any
tokenizer's stage timing looks anomalous relative to the others despite clean
VRAM readings, check whether another job was running on the GPU at the same
time before treating the timing as informative.
