import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_TRAIN_SCRIPT = "step4_train_kcbert_region_classifier.py"
DEFAULT_OUTPUT_DIR = "./smoke_kcbert_region_classifier"
DEFAULT_CACHE_DIR = "./smoke_region_classification_data"


def run_smoke_test(args: argparse.Namespace) -> None:
    train_script = Path(args.train_script)
    if not train_script.exists():
        raise FileNotFoundError(f"학습 스크립트를 찾을 수 없습니다: {train_script}")

    command = [
        sys.executable,
        str(train_script),
        "--overwrite_cache",
        "--overwrite_output_dir",
        "--output_dir",
        args.output_dir,
        "--cache_dir",
        args.cache_dir,
        "--max_train_samples",
        str(args.max_train_samples),
        "--max_eval_samples",
        str(args.max_eval_samples),
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--train_batch_size",
        str(args.train_batch_size),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--max_length",
        str(args.max_length),
        "--eval_strategy",
        "steps",
        "--save_strategy",
        "steps",
        "--eval_steps",
        str(args.eval_steps),
        "--save_steps",
        str(args.save_steps),
        "--logging_steps",
        str(args.logging_steps),
        "--save_total_limit",
        "1",
        "--preprocessing_num_workers",
        str(args.preprocessing_num_workers),
        "--dataloader_num_workers",
        str(args.dataloader_num_workers),
    ]

    if args.fp16:
        command.append("--fp16")
    if args.bf16:
        command.append("--bf16")
    if args.tokenizer_mode:
        command.extend(["--tokenizer_mode", args.tokenizer_mode])

    print("\n--- KcBERT 지역 분류 smoke test 실행 ---")
    print(" ".join(command))
    print()

    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    print("\n✅ smoke test 완료")
    print(f"출력 경로: {Path(args.output_dir).resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KcBERT 사투리 지역 분류 fine-tuning 코드가 서버에서 끝까지 동작하는지 빠르게 확인합니다."
    )
    parser.add_argument("--train_script", type=str, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)

    parser.add_argument("--max_train_samples", type=int, default=200)
    parser.add_argument("--max_eval_samples", type=int, default=50)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=64)

    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    parser.add_argument("--tokenizer_mode", type=str, choices=["dialect", "kcbert"], default="dialect")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_smoke_test(parse_args())
