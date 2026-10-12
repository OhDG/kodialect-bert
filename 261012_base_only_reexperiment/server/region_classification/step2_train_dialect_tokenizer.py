import argparse
from pathlib import Path

from tokenizers import BertWordPieceTokenizer


def train_tokenizer(args: argparse.Namespace) -> None:
    corpus_path = Path(args.corpus)
    output_dir = Path(args.output_dir)
    vocab_path = output_dir / "vocab.txt"
    if vocab_path.is_file() and not args.overwrite:
        print(f"[SKIP] Dialect tokenizer already exists: {vocab_path}")
        return
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Training corpus not found: {corpus_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False,
        lowercase=False,
    )
    print("--- Dialect WordPiece tokenizer training start ---")
    print(f"corpus: {corpus_path}")
    print(f"vocab_size: {args.vocab_size:,}")
    tokenizer.train(
        files=[str(corpus_path)],
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        wordpieces_prefix="##",
        limit_alphabet=args.limit_alphabet,
    )
    tokenizer.save_model(str(output_dir))

    actual_size = sum(1 for _ in vocab_path.open("r", encoding="utf-8"))
    if actual_size != args.vocab_size:
        raise RuntimeError(f"Expected {args.vocab_size:,} tokens, got {actual_size:,}")
    print(f"[OK] Dialect tokenizer saved: {vocab_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the dialect WordPiece tokenizer on Train only.")
    parser.add_argument("--corpus", default="./data/corpus/dialect_train_corpus.txt")
    parser.add_argument("--output_dir", default="./dialect_bert_tokenizer")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--min_frequency", type=int, default=5)
    parser.add_argument("--limit_alphabet", type=int, default=6000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train_tokenizer(parse_args())

