from tokenizers import BertWordPieceTokenizer
from pathlib import Path
import argparse
import os


def train_bert_tokenizer(args):
    corpus_file = Path(args.corpus)
    save_dir = Path(args.save_dir)

    if not corpus_file.exists():
        raise FileNotFoundError(
            f"학습 corpus 파일을 찾을 수 없습니다: {corpus_file}\n"
            f"먼저 step1_prepare_data_final.py를 실행해 dialect_train_corpus.txt를 생성하세요."
        )

    save_dir.mkdir(parents=True, exist_ok=True)

    print("--- 사투리 WordPiece 토크나이저 학습 시작 ---")
    print(f"학습 corpus: {corpus_file}")
    print(f"저장 경로: {save_dir}")
    print(f"vocab_size={args.vocab_size}, min_frequency={args.min_frequency}, limit_alphabet={args.limit_alphabet}")

    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False,
        lowercase=False,
    )

    tokenizer.train(
        files=[str(corpus_file)],
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        wordpieces_prefix="##",
        limit_alphabet=args.limit_alphabet,
    )

    tokenizer.save_model(str(save_dir))
    print(f"--- 토크나이저 학습 완료: {save_dir / 'vocab.txt'} ---")


def parse_args():
    parser = argparse.ArgumentParser(description="train corpus로 사투리 WordPiece tokenizer를 학습합니다.")
    parser.add_argument("--corpus", type=str, default="dialect_train_corpus.txt")
    parser.add_argument("--save_dir", type=str, default="./dialect_bert_tokenizer")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--min_frequency", type=int, default=5)
    parser.add_argument("--limit_alphabet", type=int, default=6000)
    return parser.parse_args()


if __name__ == "__main__":
    train_bert_tokenizer(parse_args())
