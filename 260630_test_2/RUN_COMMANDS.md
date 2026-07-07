# 260630_test_2: Dialect Tokenizer + KcBERT Continued MLM

This folder is for the stronger comparison:

```text
Dialect tokenizer
+ KcBERT encoder
+ continued MLM pretraining on dialect corpus
+ region classifier fine-tuning
```

Expected files on the server:

```text
260630_test_2/
  dialect_bert_tokenizer/vocab.txt
  step1_continue_mlm_pretrain.py
  step2_finetune_region_classifier.py
```

The scripts default to reading corpus/TSV files from:

```text
../260630_test_1/dialect_train_corpus.txt
../260630_test_1/dialect_eval_corpus.txt
../260630_test_1/region_classification_data/dialect_region_train.tsv
../260630_test_1/region_classification_data/dialect_region_eval.tsv
```

## 0. Smoke Test

Run a tiny end-to-end check first.

```bash
cd /data/ohdg/kodialect-bert/260630_test_2

python step1_continue_mlm_pretrain.py \
  --fp16 \
  --max_train_samples 1000 \
  --max_eval_samples 200 \
  --max_steps 20 \
  --eval_strategy steps \
  --save_strategy steps \
  --eval_steps 10 \
  --save_steps 10 \
  --logging_steps 5 \
  --output_dir ./smoke_kcbert_dialect_tokenizer_mlm \
  --overwrite_output_dir

python step2_finetune_region_classifier.py \
  --mlm_model_dir ./smoke_kcbert_dialect_tokenizer_mlm/final_model \
  --fp16 \
  --max_train_samples 1000 \
  --max_eval_samples 200 \
  --num_train_epochs 1 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --output_dir ./smoke_kcbert_dialect_mlm_region_classifier \
  --overwrite_output_dir
```

## 1. Continued MLM Pretraining

Full 1-epoch version:

```bash
python step1_continue_mlm_pretrain.py \
  --fp16 \
  --train_batch_size 32 \
  --eval_batch_size 64 \
  --num_train_epochs 1 \
  --output_dir ./kcbert_dialect_tokenizer_mlm \
  --overwrite_output_dir
```

If this is too slow, use fixed steps first:

```bash
python step1_continue_mlm_pretrain.py \
  --fp16 \
  --train_batch_size 32 \
  --eval_batch_size 64 \
  --max_steps 50000 \
  --eval_strategy steps \
  --save_strategy steps \
  --eval_steps 5000 \
  --save_steps 5000 \
  --logging_steps 500 \
  --output_dir ./kcbert_dialect_tokenizer_mlm_50k \
  --overwrite_output_dir
```

## 2. Region Classification Fine-Tuning

After MLM pretraining:

```bash
python step2_finetune_region_classifier.py \
  --mlm_model_dir ./kcbert_dialect_tokenizer_mlm/final_model \
  --fp16 \
  --train_batch_size 32 \
  --eval_batch_size 64 \
  --num_train_epochs 1 \
  --output_dir ./kcbert_dialect_mlm_region_classifier_epoch1 \
  --overwrite_output_dir
```

For a 50k-step MLM checkpoint:

```bash
python step2_finetune_region_classifier.py \
  --mlm_model_dir ./kcbert_dialect_tokenizer_mlm_50k/final_model \
  --fp16 \
  --train_batch_size 32 \
  --eval_batch_size 64 \
  --num_train_epochs 1 \
  --output_dir ./kcbert_dialect_mlm_50k_region_classifier_epoch1 \
  --overwrite_output_dir
```

## 3. Important Outputs

Classification outputs:

```text
eval_metrics_simple.json
eval_classification_report.json
eval_confusion_matrix.csv
final_model/
```

