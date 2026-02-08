import torch
from transformers import BertForSequenceClassification, BertTokenizer

def inference():
    model_path = "./my_dialect_bert_final"
    tokenizer_path = "./dialect_bert_tokenizer"
    
    # 1. 모델 및 토크나이저 로드
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 라벨 매핑
    labels = {0: "강원도", 1: "경상도", 2: "전라도", 3: "제주도", 4: "충청도"}

    # 지역별 10문장씩 (총 50문장)
    test_sentences = [
        # --------------------
        # 전라도 (2)
        # --------------------
        "아따 거시기하구마잉!",
        "밥은 묵었당가, 아직이면 같이 묵자잉.",
        "그라믄 안 되제잉, 다시 해봐야 쓰겄네.",
        "오늘 날씨가 참 좋구만잉.",
        "그 사람 참말로 성실허다잉.",
        "거시기 좀 가져와봐라잉.",
        "그거는 내가 해줄랑께 걱정 말어잉.",
        "어째 그리 바쁘다냐잉.",
        "아따말이여, 이거 맛있구마잉.",
        "그라제, 그게 맞다잉.",

        # --------------------
        # 경상도 (1)
        # --------------------
        "머라카노? 니 지금 머라캤나?",
        "밥 묵었나, 아직 안 묵었으면 같이 가자.",
        "그거 내 보고 하라카이.",
        "오늘 날씨 진짜 덥다 아이가.",
        "니 그 말 진짜 맞나? 다시 말해봐라.",
        "와 이래 늦었노, 다 기다렸다.",
        "그 사람 성격 참 시원시원하데이.",
        "지금 가면 시간 딱 맞다 카더라.",
        "그런 거는 신경 쓰지 마라.",
        "니 오늘 뭐 할 끼고?",

        # --------------------
        # 제주도 (3)
        # --------------------
        "제주도게난 하영 옵서예.",
        "혼저 옵서예, 차 한 잔 헙써.",
        "제주도게난 바람이 하영 불어옵서예.",
        "오늘은 비가 와서 길이 질척허우다.",
        "그 사람 말이 참 정겹수다.",
        "하영 먹어도 배가 안 부르네.",
        "이거는 제주도에서만 나는 거우다.",
        "바당에 물질하러 가야 헙서.",
        "날이 좋아서 사람들 하영 옵서예.",
        "그거는 혼디 하면 안 되주게.",

        # --------------------
        # 충청도 (4)
        # --------------------
        "기여? 아니면 말고유.",
        "그거는 좀 있다가 해도 되지유.",
        "오늘 날씨가 괜찮은 것 같어유.",
        "밥은 드셨어유, 아직이면 같이 가유.",
        "그 사람 성격이 참 느긋해유.",
        "뭐 급할 건 없잖유.",
        "그거 그렇게 하면 안 되는디유.",
        "천천히 해도 괜찮아유.",
        "그 말이 맞는 것 같기도 해유.",
        "아유, 그럴 수도 있지유.",

        # --------------------
        # 강원도 (0)
        # --------------------
        "강원도래요. 감자 먹어봤나?",
        "감자 삶아놨으니 와서 먹고 가래요.",
        "거기 눈이 와서 길이 미끄럽다우.",
        "오늘은 날씨가 참 춥수다.",
        "그 사람 참 순하우, 말도 잘 듣고.",
        "밥은 먹었수? 아직이면 같이 먹자우.",
        "이거 그냥 대충 하면 안 되래요.",
        "산 넘어 가려면 시간이 좀 걸리우.",
        "감자전 부쳐놨으니 얼른 오라우.",
        "어제는 바람이 어찌나 불던지 혼났수다.",
    ]

    print("\n--- 사투리 판별 테스트 ---")
    with torch.no_grad():
        for text in test_sentences:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding="max_length"
            ).to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()

            print(f"입력: {text}")
            print(f"결과: {labels[prediction]}\n")

if __name__ == "__main__":
    inference()
