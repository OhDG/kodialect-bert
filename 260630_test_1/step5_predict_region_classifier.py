import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL_DIR = "./smoke_kcbert_region_classifier/final_model"
DEFAULT_LABELS = ["강원도", "경상도", "전라도", "제주도", "충청도"]


def load_label(model, label_id: int) -> str:
    id2label = getattr(model.config, "id2label", None) or {}
    label = id2label.get(label_id, id2label.get(str(label_id)))
    if label is not None:
        return str(label)
    if 0 <= label_id < len(DEFAULT_LABELS):
        return DEFAULT_LABELS[label_id]
    return str(label_id)


def predict(text: str, model_dir: str, max_length: int, top_k: int) -> None:
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"모델 경로를 찾을 수 없습니다: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)

    top_k = min(top_k, probs.numel())
    scores, indices = torch.topk(probs, k=top_k)

    pred_id = int(indices[0].item())
    pred_label = load_label(model, pred_id)

    print("\n입력 문장:")
    print(text)
    print(f"\n예측 지역: {pred_label}")
    print("\n상위 확률:")
    for score, idx in zip(scores.tolist(), indices.tolist()):
        print(f"- {load_label(model, int(idx))}: {score:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="학습된 사투리 지역 분류 모델로 문장 1개를 예측합니다.")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_text = args.text

    if not input_text:
        input_text = input("분류할 문장을 입력하세요: ").strip()

    if not input_text:
        raise ValueError("입력 문장이 비어 있습니다.")

    predict(
        text=input_text,
        model_dir=args.model_dir,
        max_length=args.max_length,
        top_k=args.top_k,
    )
