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
        # 1. 경상도: 복합 어미와 성조가 느껴지는 강한 구어체
        "머라카노? 니 지금 머라캤나? 아까맨치로 가만히 있어라 좀!",
        "내는 니가 글마한테 머라 했는지 다 안다. 괘안나?",

        # 2. 전라도: 거시기와 고유 형용사/부사 테스트
        "아따 거시기하구마잉, 오늘따라 날씨가 허벌나게 포근하니 좋소잉.",
        "글제, 근디 니는 왜 그라고 있냐? 언능 오랑께!",

        # 3. 제주도: 가장 난이도 높은 고유 어휘 (표준어와 괴리가 큰 단어들)
        "무사 마씀? 어드레 감수광? 혼저 옵서예.",
        "맨도롱하냐? 고랑 몰라 봐사 알주. 제주도게난 하영 옵서.",

        # 4. 강원도: 특유의 종결 어미 (~나, ~여, ~드래요)
        "강원도래요. 옥시기 좀 먹어보드래요. 참말로 맛나드래요.",
        "니 어디 가나? 산너머 저쪽으로 가나?",

        # 5. 충청도: 느릿하고 유들유들한 말투와 반어법
        "기여? 아니면 말고. 근디 그거 좀 빨리 해주면 안 되겠슈?",
        "개 갈 안 나네. 거 좀 비켜봐유, 일 좀 하게."
    ]

    print("\n--- 토크나이징 테스트 추가 ---")
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