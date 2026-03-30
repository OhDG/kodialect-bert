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

    # 추가로 붙일 디렉토리
    extra_region_dirs = {
        "강원도": [
            "강원도_01_1인발화_따라말하기",
            "강원도_02_1인발화_질문에답하기",
            "강원도_03_2인발화"
        ],
        "경상도": [
            "경상도_01_1인발화_따라말하기",
            "경상도_02_1인발화_질문에답하기",
            "경상도_03_2인발화"
        ]
    }

    region_label = {
        "강원도": 0,
        "경상도": 1,
        "전라도": 2,
        "제주도": 3,
        "충청도": 4
    }

    file_dir = "../../project1_dataset"
    output_file = "train_data_full.csv"

    stats = {region: 0 for region in region_dirs}
    total_count = 0

    print("--- 대용량 라벨링 시작 ---")

    with open(output_file, "w", encoding="utf-8-sig", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["text", "label"])

        # -----------------------------
        # 1. 기존 데이터 유지
        # -----------------------------
        for region, subdir in region_dirs.items():
            dir_path = os.path.join(file_dir, subdir)
            if not os.path.exists(dir_path):
                print(f"⚠️ 경로 없음: {dir_path}")
                continue

            print(f"\n[{region}] 기존 데이터 처리 중...")
            json_files = [f for f in os.listdir(dir_path) if f.endswith(".json")]

            for filename in tqdm(json_files, desc=f"{region} 기존 추출 중"):
                file_path = os.path.join(dir_path, filename)
                try:
                    with open(file_path, "r", encoding="utf-8-sig") as f:
                        data = json.load(f)

                        for u in data.get("utterance", []):
                            text = u.get("dialect_form", "")

                            if isinstance(text, str) and text.strip():
                                cleaned_text = re.sub(r'\([^)]*\)|\[[^\]]*\]', '', text)
                                cleaned_text = re.sub(r'[^가-힣a-zA-Z0-9.,?! ]', '', cleaned_text)
                                cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

                                if cleaned_text:
                                    writer.writerow([cleaned_text, region_label[region]])
                                    stats[region] += 1
                                    total_count += 1

                except Exception as e:
                    print(f"\n⚠️ 파일 오류 발생: {filename} - {e}")

        # -----------------------------
        # 2. 강원도/경상도 추가 데이터 append
        # -----------------------------
        for region, subdirs in extra_region_dirs.items():
            print(f"\n[{region}] 추가 데이터 처리 중...")

            for subdir in subdirs:
                dir_path = os.path.join(file_dir, subdir)
                if not os.path.exists(dir_path):
                    print(f"⚠️ 경로 없음: {dir_path}")
                    continue

                json_files = [f for f in os.listdir(dir_path) if f.endswith(".json")]

                for filename in tqdm(json_files, desc=f"{region}-{subdir} 추가 추출 중"):
                    file_path = os.path.join(dir_path, filename)
                    try:
                        with open(file_path, "r", encoding="utf-8-sig") as f:
                            data = json.load(f)

                            text = data.get("transcription", {}).get("dialect", "")

                            if isinstance(text, str) and text.strip():
                                cleaned_text = re.sub(r'\([^)]*\)|\[[^\]]*\]', '', text)
                                cleaned_text = re.sub(r'[^가-힣a-zA-Z0-9.,?! ]', '', cleaned_text)
                                cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

                                if cleaned_text:
                                    writer.writerow([cleaned_text, region_label[region]])
                                    stats[region] += 1
                                    total_count += 1

                    except Exception as e:
                        print(f"\n⚠️ 파일 오류 발생: {filename} - {e}")

    print("\n" + "=" * 50)
    print(f"{'지역':<10} | {'문장 수':>15}")
    print("-" * 50)
    for region, count in stats.items():
        print(f"{region:<10} | {count:>15,}")
    print("-" * 50)
    print(f"{'합계':<10} | {total_count:>15,}")
    print("=" * 50)
    print(f"✅ 전체 데이터 저장 완료: {output_file}")

if __name__ == "__main__":
    prepare_full_data()