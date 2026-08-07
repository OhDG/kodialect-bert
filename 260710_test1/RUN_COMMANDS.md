# 260710_test1: accuracy-focused classifier comparison

Goal: widen the Accuracy comparison between the dialect tokenizer and KLUE tokenizer.

This experiment does not rerun MLM pretraining. It reuses the stronger 260709 epoch3 MLM models:

- `../260709_test_1/dialect_small_bert_mlm_epoch3_continued/final_model`
- `../260709_test_1/klue_small_bert_mlm_epoch3_continued/final_model`

Compared with 260709:

- No class weighting: `--class_weighting none`
- Best checkpoint is selected by `accuracy`
- Classifier fine-tuning is extended to 3 epochs
- Same train/eval TSV and same classifier settings for both tokenizers

## Run

```bash
cd /d/documents/project_1/git/kodialect-bert-local-runs/260710_test1
source ../venv/Scripts/activate

python step1_finetune_region_classifier_accuracy.py \
  --mlm_model_dir ../260709_test_1/dialect_small_bert_mlm_epoch3_continued/final_model \
  --fp16 \
  --train_batch_size 96 \
  --eval_batch_size 192 \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --class_weighting none \
  --eval_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end \
  --output_dir ./dialect_small_bert_region_classifier_epoch3_accuracy \
  --overwrite_output_dir \
&& \
python step1_finetune_region_classifier_accuracy.py \
  --mlm_model_dir ../260709_test_1/klue_small_bert_mlm_epoch3_continued/final_model \
  --fp16 \
  --train_batch_size 96 \
  --eval_batch_size 192 \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --class_weighting none \
  --eval_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end \
  --output_dir ./klue_small_bert_region_classifier_epoch3_accuracy \
  --overwrite_output_dir \
&& \
python step2_compare_accuracy_results.py
```

If CUDA OOM occurs, change every `96` to `64` and every `192` to `128`.

Expected runtime on the local RTX 4070 Ti is roughly 4-6 hours because this run skips MLM and only trains the two classifiers.
