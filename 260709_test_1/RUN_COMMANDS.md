# 260709_test_1: stronger tokenizer comparison

Goal: make the tokenizer effect easier to observe while keeping the comparison fair.

This run reuses the 260708 1-epoch MLM models and continues MLM pretraining for 2 more epochs.
That gives roughly a 3-epoch MLM model without throwing away the completed epoch1 training.

Compared with 260708:

- MLM: continue from each tokenizer's epoch1 MLM model, then train 2 more epochs.
- Classifier: train for 2 epochs.
- Classifier loss: use class-weighted cross entropy to reduce region imbalance.
- Primary metric: Macro F1, not only Accuracy.

## Local setup

Run from the local-runs directory:

```bash
cd /d/documents/project_1/git/kodialect-bert-local-runs/260709_test_1
source ../venv/Scripts/activate
```

The scripts assume these shared files already exist:

- `../shared/dialect_bert_tokenizer/vocab.txt`
- `../shared/corpus/dialect_train_corpus.txt`
- `../shared/corpus/dialect_eval_corpus.txt`
- `../shared/region_classification_data/dialect_region_train.tsv`
- `../shared/region_classification_data/dialect_region_eval.tsv`

They also reuse these 260708 outputs:

- `../260708_test_1/dialect_small_bert_mlm_epoch1/final_model`
- `../260708_test_1/klue_small_bert_mlm_epoch1/final_model`

## Smoke test

```bash
python step1_continue_small_bert_mlm.py \
  --tokenizer_mode dialect \
  --init_model_dir ../260708_test_1/dialect_small_bert_mlm_epoch1/final_model \
  --fp16 \
  --max_train_samples 1000 \
  --max_eval_samples 200 \
  --num_train_epochs 1 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --output_dir ./smoke_dialect_mlm_continued \
  --overwrite_output_dir \
&& \
python step2_finetune_region_classifier_weighted.py \
  --mlm_model_dir ./smoke_dialect_mlm_continued/final_model \
  --fp16 \
  --max_train_samples 1000 \
  --max_eval_samples 200 \
  --num_train_epochs 1 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --class_weighting balanced \
  --output_dir ./smoke_dialect_region_classifier_weighted \
  --overwrite_output_dir
```

## Full run: recommended amplified comparison

This is the main command. It uses batch 96 to speed up the 4070 Ti run. If CUDA OOM occurs, rerun with `--train_batch_size 64 --eval_batch_size 128`.

```bash
python step1_continue_small_bert_mlm.py \
  --tokenizer_mode dialect \
  --init_model_dir ../260708_test_1/dialect_small_bert_mlm_epoch1/final_model \
  --fp16 \
  --train_batch_size 96 \
  --eval_batch_size 192 \
  --num_train_epochs 2 \
  --learning_rate 3e-5 \
  --warmup_ratio 0.03 \
  --output_dir ./dialect_small_bert_mlm_epoch3_continued \
  --overwrite_output_dir \
&& \
python step2_finetune_region_classifier_weighted.py \
  --mlm_model_dir ./dialect_small_bert_mlm_epoch3_continued/final_model \
  --fp16 \
  --train_batch_size 96 \
  --eval_batch_size 192 \
  --num_train_epochs 2 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --class_weighting balanced \
  --output_dir ./dialect_small_bert_region_classifier_epoch2_weighted \
  --overwrite_output_dir \
&& \
python step1_continue_small_bert_mlm.py \
  --tokenizer_mode klue \
  --init_model_dir ../260708_test_1/klue_small_bert_mlm_epoch1/final_model \
  --fp16 \
  --train_batch_size 96 \
  --eval_batch_size 192 \
  --num_train_epochs 2 \
  --learning_rate 3e-5 \
  --warmup_ratio 0.03 \
  --output_dir ./klue_small_bert_mlm_epoch3_continued \
  --overwrite_output_dir \
&& \
python step2_finetune_region_classifier_weighted.py \
  --mlm_model_dir ./klue_small_bert_mlm_epoch3_continued/final_model \
  --fp16 \
  --train_batch_size 96 \
  --eval_batch_size 192 \
  --num_train_epochs 2 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --class_weighting balanced \
  --output_dir ./klue_small_bert_region_classifier_epoch2_weighted \
  --overwrite_output_dir \
&& \
python step3_compare_tokenizer_results.py
```

## Why these settings

- `--init_model_dir`: reuses the finished 260708 MLM epoch1 models.
- `--num_train_epochs 2` for MLM: adds enough language-model learning for tokenizer differences to emerge.
- `--class_weighting balanced`: reduces majority-region dominance and makes Macro F1 more meaningful.
- `--num_train_epochs 2` for classifier: gives the classifier more time to learn minority-region patterns.
- `--train_batch_size 96`: uses more of local GPU memory than the previous batch 64 run.

Expected runtime on the local RTX 4070 Ti is roughly 6-8 hours. If batch 96 causes OOM, use batch 64 and expect closer to 8-10 hours.
