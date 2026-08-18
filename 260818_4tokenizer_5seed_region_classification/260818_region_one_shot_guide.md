# 260818 A6000 one-command runner (project-local setup guide)

## 1) 서버에서 올릴 파일
1. `run_260818_region_one_shot.py` (아래 스크립트)만 올리면 됨.
2. 실행 디렉토리에는 기존 실험 스크립트 2개가 있어야 함:
   - `step1_continue_mlm_pretrain.py`
   - `step2_finetune_region_classifier.py`

권장 실험 루트:
`<repo_root>/260818_4tokenizer_5seed_region_classification_a6000`

## 2) 서버에서 한 번에 실행(가장 권장)
```bash
mkdir -p /path/to/repo/260818_4tokenizer_5seed_region_classification_a6000
cd /path/to/repo/260818_4tokenizer_5seed_region_classification_a6000

# code_root는 step1/step2.py가 있는 경로로 지정
python run_260818_region_one_shot.py \
  --code_root /path/to/repo/260807_4tokenizer_5seed_region_classification \
  --experiment_root /path/to/repo/260818_4tokenizer_5seed_region_classification_a6000 \
  --tokenizers dialect,klue,kobert,mbert \
  --seeds 13,21,42,87,100 \
  --gpu_id 0 \
  --stream_logs
```

`--no_stream_logs`으로 바꾸면 터미널 출력은 줄일 수 있습니다.

## 3) 백그라운드 실행
```bash
nohup python run_260818_region_one_shot.py \
  --code_root /path/to/repo/260807_4tokenizer_5seed_region_classification \
  --experiment_root /path/to/repo/260818_4tokenizer_5seed_region_classification_a6000 \
  --tokenizers dialect,klue,kobert,mbert \
  --seeds 13,21,42,87,100 \
  --gpu_id 0 \
  > /path/to/repo/260818_4tokenizer_5seed_region_classification_a6000/run.log 2>&1 &
```

실행 중단:
`tail -f run.log`

## 4) 출력 파일/로그
`logs` under each seed:
- `.../{tokenizer}/seed_{seed}/logs/mlm_a1.log`
- `.../{tokenizer}/seed_{seed}/logs/classifier_a1.log`
- `.../{tokenizer}/seed_{seed}/logs/mlm_a1_gpu.csv` (해당 PID GPU VRAM/전체 GPU util)
- `results/run_summary.csv` (성공/실패, 재시도, 피크 VRAM 등 요약)

## 5) GPU 모니터링이 내 프로세스만 되는 이유
`nvidia-smi`에서 PID를 받아서, 해당 PID의 `used_gpu_memory`만 수집하도록 설계됨.
또한 실시간 출력/로그로 전체 로그를 남기면서 오직 실행한 Python 프로세스 자식 기준으로 모니터링.
