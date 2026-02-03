import os
import json
import re
import csv
from tqdm import tqdm

def prepare_full_data():
    region_dirs = {
        "강원도": "JSON만_모은폴더_강원도",
        "경상도": "JSON만_모은폴더_경상도",
        "전라도": "JSON만_모은폴더_전라도",
        "제주도": "JSON만_모은폴더_제주도",
        "충청도": "JSON만_모은폴더_충청도"
    }
    region_label = {"강원도": 0, "경상도": 1, "전라도": 2, "제주도": 3, "충청도": 4}
    file_dir = "../../project1_dataset"
    output_file = "train_data_full.csv"
    
    print(f"--- 670만개 대용량 라벨링 시작 ---")
    
    with open(output_file, "w", encoding="utf-8-sig", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["text", "label"]) # 헤더 작성

        for region, subdir in region_dirs.items():
            dir_path = os.path.join(file_dir, subdir)
            if not os.path.exists(dir_path): continue
            
            json_files = [f for f in os.listdir(dir_path) if f.endswith(".json")]
            
            for filename in tqdm(json_files, desc=f"{region} 추출 중"):
                file_path = os.path.join(dir_path, filename)
                try:
                    with open(file_path, "r", encoding="utf-8-sig") as f:
                        data = json.load(f)
                        for u in data.get("utterance", []):
                            text = u.get("dialect_form", "")
                            if isinstance(text, str) and text.strip():
                                # 정규식 처리
                                cleaned_text = re.sub(r'\([^)]*\)|\[[^)]*\]', '', text)
                                cleaned_text = re.sub(r'[^가-힣a-zA-Z0-9.,?! ]', '', cleaned_text).strip()
                                if cleaned_text:
                                    writer.writerow([cleaned_text, region_label[region]])
                except Exception: continue

    print(f"✅ 전체 데이터 저장 완료: {output_file}")

if __name__ == "__main__":
    prepare_full_data()