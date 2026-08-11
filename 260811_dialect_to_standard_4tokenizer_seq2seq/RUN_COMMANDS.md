# 260811 Dialect-to-standard experiment

## Design

- Reuse the exact file-level 80/10/10 split and four source MLM encoders from `260807_4tokenizer_5seed_region_classification`.
- Read `dialect_form` as the source and `standard_form` as the target.
- Pretrain one shared KLUE-tokenized standard-form causal decoder.
- Fine-tune `dialect`, `klue`, `kobert`, and `mbert` source encoders with seeds `13, 21, 42, 87, 100`.
- Select each run's best checkpoint by Validation chrF++, then report independent Test metrics.
- Report overall, changed-only, identity-only, and per-region generation metrics.
- Record target-process VRAM/SM samples separately from whole-device telemetry.

The source MLM models are intentionally reused. Their previous device-level GPU logs are not reused as process-only measurements. This experiment measures the new shared decoder and translation stages directly.

## Server installation

```bash
cd /data/ohdg/kodialect-bert/260811_dialect_to_standard_4tokenizer_seq2seq
python -m pip install -r requirements_260811.txt
```

## Full RTX A6000 experiment

The command streams all logs to the current terminal and also writes stage logs under `logs/`.

```bash
python run_full_experiment.py
```

Resume after an interruption with the same command. Completed stages are skipped and an incomplete Trainer stage resumes from its latest checkpoint.

To discard existing experiment outputs and run every stage again:

```bash
python run_full_experiment.py --overwrite
```

## Local end-to-end smoke test

Run this from the same directory in the local CUDA environment:

```bash
python run_full_experiment.py --smoke
```

## Important outputs

```text
results/final_results.md
results/final_results.json
results/test_summary.csv
logs/*_gpu.csv
logs/*_gpu_summary.json
outputs/translation/<tokenizer>/seed_<seed>/test_generation_report.json
outputs/translation/<tokenizer>/seed_<seed>/test_predictions.tsv.gz
```

`process_*` GPU fields target only the child training PID. `device_*` fields describe the whole GPU and can include other users. PyTorch peak allocated/reserved VRAM is measured inside each training process.
