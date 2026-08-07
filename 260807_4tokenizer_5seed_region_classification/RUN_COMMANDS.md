# 260807 four-tokenizer, five-seed region classification

## Experiment design

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
cd /data/ohdg/kodialect-bert/260807_4tokenizer_5seed_region_classification
```

The default source manifest is:

```text
../260630_test_1/corpus_split_manifest.csv
```

Its existing JSON paths are reused to generate the new split.

## One-time dependency and smoke check

```bash
python -m pip install -r requirements_260807.txt
python run_full_experiment.py --smoke
```

The smoke command still prepares the 80/10/10 data when it does not already exist. Subsequent runs reuse it.

## Full experiment

```bash
python run_full_experiment.py
```

The runner is resumable. Re-entering the same command skips completed runs and resumes the latest complete
Trainer checkpoint for an interrupted stage.

For a persistent visible terminal session:

```bash
tmux new -s dialect_260807
python run_full_experiment.py
```

Detach with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t dialect_260807
```

## A6000 profile

| Stage | Tokenizer | Micro batch | Accumulation | Effective batch | Eval batch |
| --- | --- | ---: | ---: | ---: | ---: |
| MLM | dialect/KLUE/KoBERT | 256 | 1 | 256 | 512 |
| MLM | mBERT | 128 | 2 | 256 | 256 |
| Classification | all | 256 | 1 | 256 | 2,048 |

Common settings: maximum length 128, FP16, TF32, fused AdamW, 8 DataLoader workers, and 16 tokenizer workers.

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
```

Each GPU stage records average utilization, peak VRAM, average power, estimated energy, maximum temperature,
and wall-clock time in `logs/*_gpu_summary.json`.

## Important interpretation

The primary controlled comparison is dialect versus KLUE because both use a 32,000-token vocabulary.
KoBERT and mBERT are native-vocabulary reference tokenizers; their native vocabulary sizes change the embedding and MLM
output parameter counts. All four receive equal MLM epochs, classification seeds, data, and effective batch sizes.
