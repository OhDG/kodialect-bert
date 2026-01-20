from tokenizers import BertWordPieceTokenizer
import os

def train_bert_tokenizer():
    corpus_file = "dialect_corpus.txt"
    save_dir = "./dialect_bert_tokenizer"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 토크나이저 초기화
    # clean_text: 텍스트 정규화
    # handle_chinese_chars: 한자 처리 여부
    # strip_accents: 악센트 제거 (한국어는 False 권장)
    # lowercase: 소문자화 여부
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False, 
        lowercase=False
    )

    # 학습 시작
    tokenizer.train(
        files=[corpus_file],
        vocab_size=32000, # 보통 BERT는 3만 내외 사용
        min_frequency=2,  # 최소 2번 이상 등장해야 어휘 사전에 추가
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        wordpieces_prefix="##",
        limit_alphabet=6000
    )

    # 저장 (vocab.txt 파일이 생성됨)
    tokenizer.save_model(save_dir)
    print(f"--- 토크나이저 학습 완료 및 {save_dir}에 저장됨 ---")

if __name__ == "__main__":
    train_bert_tokenizer()