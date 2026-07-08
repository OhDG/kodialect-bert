# 260708_test_1: tokenizer-centered small BERT experiments

This experiment compares tokenizers under the same model family and training flow:

1. Dialect tokenizer + small BERT MLM from scratch + region classification
2. KLUE-BERT tokenizer + small BERT MLM from scratch + region classification

The baseline tokenizer is `klue/bert-base`, whose vocab size is 32k. This makes the comparison cleaner against the 32k dialect tokenizer.

## Expected files

Place or keep these files relative to `260708_test_1`:

- `./dialect_bert_tokenizer/vocab.txt`
- `../260630_test_1/dialect_train_corpus.txt`
- `../260630_test_1/dialect_eval_corpus.txt`
- `../260630_test_1/region_classification_data/dialect_region_train.tsv`
- `../260630_test_1/region_classification_data/dialect_region_eval.tsv`

## Smoke test: dialect tokenizer

```bash
python step1_pretrain_small_bert_mlm.py \
  --tokenizer_mode dialect \
  --fp16 \
  --max_train_samples 1000 \
  --max_eval_samples 200 \
  --num_train_epochs 1 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --output_dir ./smoke_dialect_small_bert_mlm \
  --overwrite_output_dir \
&& \
python step2_finetune_region_classifier.py \
  --mlm_model_dir ./smoke_dialect_small_bert_mlm/final_model \
  --fp16 \
  --max_train_samples 1000 \
  --max_eval_samples 200 \
  --num_train_epochs 1 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --output_dir ./smoke_dialect_small_bert_region_classifier \
  --overwrite_output_dir
```

## Smoke test: KLUE-BERT tokenizer

```bash
python step1_pretrain_small_bert_mlm.py \
  --tokenizer_mode klue \
  --klue_tokenizer_name klue/bert-base \
  --fp16 \
  --max_train_samples 1000 \
  --max_eval_samples 200 \
  --num_train_epochs 1 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --output_dir ./smoke_klue_small_bert_mlm \
  --overwrite_output_dir \
&& \
python step2_finetune_region_classifier.py \
  --mlm_model_dir ./smoke_klue_small_bert_mlm/final_model \
  --fp16 \
  --max_train_samples 1000 \
  --max_eval_samples 200 \
  --num_train_epochs 1 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --output_dir ./smoke_klue_small_bert_region_classifier \
  --overwrite_output_dir
```

## Full run: dialect tokenizer

```bash
python step1_pretrain_small_bert_mlm.py \
  --tokenizer_mode dialect \
  --fp16 \
  --train_batch_size 64 \
  --eval_batch_size 128 \
  --num_train_epochs 1 \
  --output_dir ./dialect_small_bert_mlm_epoch1 \
  --overwrite_output_dir \
&& \
python step2_finetune_region_classifier.py \
  --mlm_model_dir ./dialect_small_bert_mlm_epoch1/final_model \
  --fp16 \
  --train_batch_size 64 \
  --eval_batch_size 128 \
  --num_train_epochs 1 \
  --output_dir ./dialect_small_bert_region_classifier_epoch1 \
  --overwrite_output_dir
```

## Full run: KLUE-BERT tokenizer

```bash
python step1_pretrain_small_bert_mlm.py \
  --tokenizer_mode klue \
  --klue_tokenizer_name klue/bert-base \
  --fp16 \
  --train_batch_size 64 \
  --eval_batch_size 128 \
  --num_train_epochs 1 \
  --output_dir ./klue_small_bert_mlm_epoch1 \
  --overwrite_output_dir \
&& \
python step2_finetune_region_classifier.py \
  --mlm_model_dir ./klue_small_bert_mlm_epoch1/final_model \
  --fp16 \
  --train_batch_size 64 \
  --eval_batch_size 128 \
  --num_train_epochs 1 \
  --output_dir ./klue_small_bert_region_classifier_epoch1 \
  --overwrite_output_dir
```

## Notes

- Both runs use the same small BERT architecture by default:
  - `hidden_size=384`
  - `num_hidden_layers=6`
  - `num_attention_heads=6`
  - `intermediate_size=1536`
- The model is initialized from scratch in both runs.
- This removes the KcBERT pretrained tokenizer/embedding advantage and makes the comparison more tokenizer-centered.
