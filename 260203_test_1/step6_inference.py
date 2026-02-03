import torch
from transformers import BertForSequenceClassification, BertTokenizer

def inference():
    model_path = "./my_dialect_bert_model"
    tokenizer_path = "./dialect_bert_tokenizer"
    
    # 1. 모델 및 토크나이저 로드
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 라벨 매핑
    labels = {0: "강원도", 1: "경상도", 2: "전라도", 3: "제주도", 4: "충청도"}

    test_sentences = [
        "아따 거시기하구마잉!", # 전라도 예상
        "머라카노? 니 지금 머라캤나?", # 경상도 예상
        "제주도게난 하영 옵서예.", # 제주도 예상
        "기여? 아니면 말고유.", # 충청도 예상
        "강원도래요. 감자 먹어봤나?" # 강원도 예상
    ]

    print("\n--- 사투리 판별 테스트 ---")
    with torch.no_grad():
        for text in test_sentences:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length").to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()
            
            print(f"입력: {text}")
            print(f"결과: {labels[prediction]}\n")

if __name__ == "__main__":
    inference()