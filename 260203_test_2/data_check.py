import pandas as pd

# 전체 데이터 파일 로드 (시간이 좀 걸릴 수 있습니다)
df = pd.read_csv("train_data_full.csv")

print("--- 지역별 데이터 개수 확인 ---")
print(df['label'].value_counts())

# 라벨 매핑: 0: 강원, 1: 경상, 2: 전라, 3: 제주, 4: 충청