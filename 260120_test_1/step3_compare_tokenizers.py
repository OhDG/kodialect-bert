import os
from transformers import BertTokenizer

def compare_tokenizers():
    # 1. 내가 만든 사투리 토크나이저 로드
    my_tokenizer_path = "./dialect_bert_tokenizer"
    my_tokenizer = BertTokenizer.from_pretrained(my_tokenizer_path, do_lower_case=False)

    # 2. Google Multilingual BERT 토크나이저 로드
    google_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    test_sentences = [
        "머라카노? 니 지금 머라캤나?",
        "아따 거시기하구마잉, 허벌나게 좋소잉.",
        "무사 마씀? 어드레 감수광? 혼저 옵서예.",
        "기여? 아니면 말고. 빨리 해주면 안 되겠슈?"
    ]

    print("="*80)
    print(f"{'입력 문장':^40}")
    print("="*80)

    for sentence in test_sentences:
        # 내 토크나이저 결과
        my_tokens = my_tokenizer.tokenize(sentence)
        
        # Google 토크나이저 결과
        google_tokens = google_tokenizer.tokenize(sentence)

        print(f"\n[입력]: {sentence}")
        print(f" ▶ My Tokenizer     : {my_tokens}")
        print(f" ▶ Google Tokenizer : {google_tokens}")
        
        # 차이점 분석 팁
        if len(my_tokens) < len(google_tokens):
            print(" 💡 분석: 내 토크나이저가 사투리 어휘를 더 의미 있는 단위로 묶어서 파악합니다.")
        print("-" * 50)

if __name__ == "__main__":
    compare_tokenizers()