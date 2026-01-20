# from transformers import BertTokenizer

# def test_tokenizer():
#     # 학습된 경로에서 토크나이저 불러오기
#     tokenizer_path = "./dialect_bert_tokenizer"
#     tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

#     test_sentences = [
#         "강원도래요, 감자 먹어봤나?",
#         "머라카노, 니 지금 머라캤나?",
#         "제주도게난 하영 옵서예"
#     ]

#     print("\n--- 토크나이징 테스트 ---")
#     for sentence in test_sentences:
#         tokens = tokenizer.tokenize(sentence)
#         ids = tokenizer.encode(sentence)
#         print(f"입력: {sentence}")
#         print(f"토큰: {tokens}")
#         print(f"ID  : {ids}\n")

# if __name__ == "__main__":
#     test_tokenizer()

from transformers import BertTokenizer

def test_tokenizer():
    # 학습된 경로
    tokenizer_path = "./dialect_bert_tokenizer"
    
    # 중요: 학습 시 lowercase=False로 했다면 여기서도 맞춰줘야 합니다.
    # 또한, 직접 만든 vocab.txt를 명시적으로 불러옵니다.
    tokenizer = BertTokenizer.from_pretrained(
        tokenizer_path, 
        do_lower_case=False,
        clean_text=True
    )

    test_sentences = [
        "강원도래요, 감자 먹어봤나?",
        "머라카노, 니 지금 머라캤나?",
        "제주도게난 하영 옵서예"
    ]

    print("\n--- 토크나이징 테스트 (수정 버전) ---")
    for sentence in test_sentences:
        # 1. 텍스트를 토큰으로 쪼개기
        tokens = tokenizer.tokenize(sentence)
        # 2. 토큰을 숫자로 변환 (CLS, SEP 포함)
        ids = tokenizer.encode(sentence)
        
        print(f"입력: {sentence}")
        print(f"토큰: {tokens}")
        print(f"ID  : {ids}")
        # 역으로 숫자를 다시 글자로 바꿔보기 (검증)
        print(f"복원: {tokenizer.decode(ids)}\n")

if __name__ == "__main__":
    test_tokenizer()