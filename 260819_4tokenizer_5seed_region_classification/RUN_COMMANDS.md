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

This directory abandons that redesign entirely. It is `260807`'s **original,
unmodified batch profile and hyperparameters** (the ones a real server-trained
run of that design has already produced usable encoders with — the same
encoders `260811_dialect_to_standard_4tokenizer_seq2seq` already fine-tunes
against). The GPU measurement method is upgraded to process-only: it now
records only this experiment's own process (via `pynvml`, ported from the
`260814_..._local` / `260818` work) instead of whole-device `nvidia-smi`
snapshots that could include other users' jobs on a shared GPU.

**MLM pretraining is skipped by default.** `260807`'s server run already
produced real, completed `outputs/mlm/<tokenizer>/final_model` encoders — the
exact ones `260811` (translation) already fine-tunes against. Retraining MLM
here would cost several more hours per tokenizer just to reproduce something
that, given CUDA's run-to-run nondeterminism, wouldn't even come out
bit-identical to the original — a fresh retrain is *less* faithful to "the same
encoder" than just reusing the real file. So by default, `run_full_experiment.py`
copies `260807`'s `final_model/` + `mlm_pretraining_metadata.json` for each
tokenizer straight into this run's `outputs/mlm/<tokenizer>/` and skips both the
dialect-tokenizer-training stage and MLM pretraining entirely — only the
classifier fine-tuning stage (4 tokenizers × 5 seeds) actually runs and gets
fresh process-only GPU measurement. This also means classification and
translation now share the literal same encoder files, not just the same design.

The one thing NOT freshly measured under reuse: the MLM row in
`results/efficiency_summary.csv` / `final_results.md` is copied from `260807`'s
original whole-device GPU log (if present there), not re-measured by this run's
process-only monitor. Note this caveat if quoting MLM-stage GPU numbers from a
reused run. To force a full from-scratch run instead (fresh MLM stage too, with
process-only measurement throughout):

```bash
python run_full_experiment.py --reuse_mlm_from ""
```

Nothing about `260811` (translation) needs to change — it already reuses real,
successfully-trained `260807` encoders and is unaffected by any of this.

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

By default this copies the four MLM encoders from `260807` (see above) and runs
only data prep + classifier fine-tuning (4 tokenizers × 5 seeds = 20 Trainer
runs) — no dialect-tokenizer-training stage, no MLM pretraining stage. The
runner is resumable regardless: re-entering the same command skips completed
runs and resumes the latest complete Trainer checkpoint for an interrupted
stage.

For a persistent visible terminal session:

```bash
tmux new -s dialect_260819
python run_full_experiment.py
```

Detach with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t dialect_260819
```

## A6000 profile (unchanged from 260807 — no redesign this time)

The MLM row only applies if you force a from-scratch run with
`--reuse_mlm_from ""`; the default reuse path never invokes the MLM script at
all, so these batch sizes only matter in that fallback case.

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

Each classifier stage records process-only peak allocated/reserved VRAM and
process SM utilization in `logs/04_classifier_*_gpu_summary.json` and in
`outputs/classifiers/.../experiment_metadata.json` (`process_gpu_memory`, from
`torch.cuda.max_memory_allocated`/`max_memory_reserved` inside the training
process itself — cannot include another user's job even on a shared card). Under
the default reuse path, `logs/03_mlm_*_gpu_summary.json` (if present) is copied
verbatim from `260807` and reflects that run's own whole-device measurement, not
this run's process-only method — see "Why this directory exists" above.

## Important interpretation

The primary controlled comparison is dialect versus KLUE because both use a 32,000-token vocabulary.
KoBERT and mBERT are native-vocabulary reference tokenizers; their native vocabulary sizes change the embedding and MLM
output parameter counts. All four receive equal MLM epochs, classification seeds, data, and effective batch sizes.

GPU measurements in this run are process-scoped (this experiment's PID and its
descendants only), unlike 260807's original whole-device `nvidia-smi` snapshots.
If another job runs on the same GPU concurrently, it cannot leak into these
numbers — verify by checking `nvml_process_vram_supported: true` in the
`*_gpu_summary.json` files after the run.
