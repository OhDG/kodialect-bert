import os
import json
import re
import pandas as pd
from tqdm import tqdm

def prepare_labeled_data(debug=True):
    region_dirs = {
        "강원도": "JSON만_모은폴더_강원도",
        "경상도": "JSON만_모은폴더_경상도",
        "전라도": "JSON만_모은폴더_전라도",
        "제주도": "JSON만_모은폴더_제주도",
        "충청도": "JSON만_모은폴더_충청도"
    }
    region_label = {"강원도": 0, "경상도": 1, "전라도": 2, "제주도": 3, "충청도": 4}
    file_dir = "../../project1_dataset"
    output_file = "train_data_sampled.csv" if debug else "train_data_full.csv"
    
    data_list = []

    print(f"--- 라벨링 데이터 준비 시작 (디버그 모드: {debug}) ---")
    for region, subdir in region_dirs.items():
        dir_path = os.path.join(file_dir, subdir)
        if not os.path.exists(dir_path): continue
        
        json_files = [f for f in os.listdir(dir_path) if f.endswith(".json")]
        
        # 테스트를 위해 지역당 100문장만 (debug=True일 때)
        count = 0
        for filename in tqdm(json_files, desc=f"{region} 추출"):
            file_path = os.path.join(dir_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                    utterances = data.get("utterance", [])
                    for u in utterances:
                        text = u.get("dialect_form", "")
                        if isinstance(text, str) and text.strip():
                            cleaned_text = re.sub(r'\([^)]*\)|\[[^)]*\]', '', text)
                            cleaned_text = re.sub(r'[^가-힣a-zA-Z0-9.,?! ]', '', cleaned_text).strip()
                            if cleaned_text:
                                data_list.append({"text": cleaned_text, "label": region_label[region]})
                                count += 1
            except Exception: continue
            
            if debug and count > 100: break # 지역당 100개만 뽑고 다음 지역으로

    df = pd.DataFrame(data_list)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"✅ {len(df)}개의 데이터를 {output_file}에 저장했습니다.")

if __name__ == "__main__":
    # 처음에는 True로 해서 코드가 도는지 확인하세요!
    prepare_labeled_data(debug=True)