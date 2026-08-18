# 260818 four-tokenizer, five-seed region classification (server rerun)

## Why this rerun exists

The 260807 experiment directory was designed for an A6000-class 48 GB GPU but was
never actually executed at full scale on one; the numbers currently drafted in the
thesis (Chapter 4) came from a follow-up local run on an RTX 4070 Ti 12 GB
(`260814_4tokenizer_5seed_region_classification_local`), scaled down to fit that
card. Two confounds were identified in that local run:

1. **mBERT-only micro-batch reduction.** `step3_pretrain_small_bert_mlm.py` had a
   `apply_low_vram_mbert_profile()` hook that halved mBERT's micro batch whenever
   total GPU memory was under 20 GB. This directory removes that function entirely
   — mBERT is handled the same way as the other three at the orchestration level
   (see "Batch profile" below), just with a smaller micro batch for a documented,
   vocabulary-size reason, not a runtime hardware detection.
2. **Allocator fragmentation on the dialect MLM run.** On the 12 GB card, the
   dialect tokenizer's MLM run reserved ~20 GB of PyTorch-cached memory despite
   allocating (using) almost the same amount as KLUE-BERT (~8.5 GB both), which
   pushed it into Windows' slow shared-GPU-memory fallback and inflated its wall
   time disproportionately. A 48 GB card removes the physical pressure that
   triggers this. (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was tried
   as an extra fragmentation-reduction measure but was dropped: this server's
   PyTorch build raises a hard error on that key instead of ignoring it, so the
   runner no longer sets it.)

Everything else — data split, preprocessing, tokenizer vocabulary size (32,000),
model architecture, epochs, learning rate, warmup, weight decay, effective batch
size (256), FP16, TF32, fused AdamW, five classification seeds — is unchanged from
what is already documented in the thesis draft.

## Server hardware profile

Target: RTX A6000 48 GB (`nvidia-smi` confirmed idle, 0 other processes).

### Batch profile

| Stage | Tokenizer | Micro batch | Accumulation | Effective batch | Eval batch |
| --- | --- | ---: | ---: | ---: | ---: |
| MLM | dialect / klue / kobert | 256 | 1 | 256 | 2,048 |
| MLM | mbert | 128 | 2 | 256 | 512 |
| Classification | all four | 256 | 1 | 256 | 4,096 |

The MLM split is not a hardware workaround: mBERT's 119,547-token vocabulary makes
the `[batch, seq_len, vocab_size]` logits tensor in the MLM head far larger than the
other three tokenizers' (32,000 or 8,002-way), so it genuinely needs a smaller
micro batch at the same effective batch size. This is defined directly in
`run_full_experiment.py` (`LARGE_VOCAB_TOKENIZERS = {"mbert"}`), applied
unconditionally regardless of detected GPU memory — there is no runtime
card-detection branch anymore. The classifier head is only 5-way (region logits),
so it does not have this asymmetry and all four tokenizers use the same batch
there.

Effective batch size (256) is identical to the local run and to the thesis's
already-drafted methodology (Table 4-5) in both stages — only the micro-batch /
accumulation split changed, which is a pure throughput lever (fewer forward/backward
calls per optimizer step) and does not alter training dynamics or require
re-tuning the learning rate.

If you want to push further (e.g. raise the effective batch itself for more raw
speed), that changes the science, not just the engineering, and would need the
learning rate re-tuned to match (linear scaling rule) — talk to me first before
doing this if you want it, since it would also require rewording Table 4-5/4-6.

### Other aggressive settings

- `--dataloader_num_workers 16` and `--preprocessing_num_workers 16` (defaults;
  check `nproc` on the server and raise if you have more cores idle).
- `--tokenize_batch_size 8000` (defaults).
- `--classifier_eval_batch_size 4096` — evaluation has no gradients/optimizer
  state, so it is much cheaper per sample than training; this is set higher than
  the training batch specifically to speed up the ~713K-sentence Test evaluation.

## Environment

```bash
cd /data/ohdg/kodialect-bert/260818_4tokenizer_5seed_region_classification
python -m pip install -r requirements_260818.txt
```

PyTorch with CUDA is assumed to already be installed in your `dialect1` environment
and is intentionally not pinned in the requirements file — verify it matches driver
580.173.02 / CUDA 13.0 before running:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Copy the exact server data (optional)

If a prepared `data/` directory and trained dialect tokenizer from 260807 already
exist on this server, reuse them instead of re-splitting:

```bash
cp -r /data/ohdg/kodialect-bert/260807_4tokenizer_5seed_region_classification/data .
cp -r /data/ohdg/kodialect-bert/260807_4tokenizer_5seed_region_classification/dialect_bert_tokenizer .
```

Otherwise the runner creates both automatically from `../260630_test_1/corpus_split_manifest.csv`
using the same 8:1:1 file-level split (seed 42) as before.

The runner verifies these six non-empty files before starting:

```text
data/corpus/dialect_{train,validation,test}_corpus.txt
data/region_classification/dialect_region_{train,validation,test}.tsv
```

## Smoke test

Checks the full pipeline and process-only GPU measurement with one tokenizer and
one seed before committing to the full run:

```bash
python run_full_experiment.py --smoke --tokenizers dialect --seeds 13 --overwrite
```

## Full experiment

```bash
python run_full_experiment.py
```

Output streams to the terminal and is also written under `logs/`. The command is
resumable: rerunning it skips completed stages (MLM per tokenizer, classifier per
tokenizer/seed) and resumes an interrupted Trainer checkpoint. Do not add
`--overwrite` to a normal resume command — it discards completed stages.

To rerun with different batch sizes without touching the script, pass overrides,
e.g.:

```bash
python run_full_experiment.py --classifier_eval_batch_size 8192 --dataloader_num_workers 32
```

## Process-only GPU measurement

- `process_gpu_memory` in every MLM/classifier metadata file comes from
  `torch.cuda.max_memory_allocated()` / `max_memory_reserved()` — allocator values
  covering only the training Python process itself.
- `gpu_monitor.py` samples only the training root PID and its descendants via
  `pynvml`. On Linux, `nvmlDeviceGetComputeRunningProcesses` typically reports real
  per-process VRAM and SM utilization (unlike Windows WDDM, which often withholds
  both) — expect `nvml_process_vram_supported: true` in the GPU summary JSONs this
  time, unlike the 260814 local run's `false`.

Final outputs:

```text
results/final_results.md
results/final_results.json
results/overall_by_seed.csv
results/overall_summary.csv
results/efficiency_summary.csv
```

## After the run

Send back (or upload) `results/final_results.md`, `results/final_results.json`,
`results/efficiency_summary.csv`, and the `logs/*_gpu_summary.json` files — Chapter
4 (Tables 4-9 through 4-11, 4-15) and Chapter 5's limitations section will be
updated from these to replace the 260814 local-run numbers.
