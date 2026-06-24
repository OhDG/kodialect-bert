import os
import json
import re
import sys
from tqdm import tqdm 

def prepare_corpus():
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

    file_dir = "../../project1_dataset"
    output_file = "dialect_corpus.txt"
    
    # 지역별 통계를 저장할 딕셔너리
    stats = {region: {"count": 0, "size": 0} for region in region_dirs}
    total_count = 0
    total_size = 0

    print("--- 대용량 데이터 추출 및 통계 분석 시작 ---")
    
    with open(output_file, "w", encoding="utf-8") as f_out:
        # -----------------------------
        # 1. 기존 코퍼스 유지
        # -----------------------------
        for region, subdir in region_dirs.items():
            dir_path = os.path.join(file_dir, subdir)
            if not os.path.exists(dir_path):
                print(f"⚠️ 경로 없음: {dir_path}")
                continue
            
            print(f"\n[{region}] 기존 코퍼스 처리 중...")
            json_files = [f for f in os.listdir(dir_path) if f.endswith(".json")]
            
            for filename in tqdm(json_files, desc=f"{region} 기존 진행도"):
                file_path = os.path.join(dir_path, filename)
                try:
                    with open(file_path, "r", encoding="utf-8-sig") as f_in:
                        data = json.load(f_in)
                        utterances = data.get("utterance", [])
                        for u in utterances:
                            text = u.get("dialect_form", "")
                            if isinstance(text, str) and text.strip():
                                cleaned_text = re.sub(r'\([^)]*\)|\[[^\]]*\]', '', text)
                                cleaned_text = re.sub(r'[^가-힣a-zA-Z0-9.,?! ]', '', cleaned_text).strip()
                                
                                if cleaned_text:
                                    output_line = cleaned_text + "\n"
                                    f_out.write(output_line)
                                    
                                    line_bytes = len(output_line.encode('utf-8'))
                                    stats[region]["count"] += 1
                                    stats[region]["size"] += line_bytes
                                    total_count += 1
                                    total_size += line_bytes
                                    
                except Exception as e:
                    print(f"\n⚠️ 파일 오류 발생: {filename} - {e}")

        # -----------------------------
        # 2. 강원도/경상도 추가 코퍼스 append
        # -----------------------------
        for region, subdirs in extra_region_dirs.items():
            print(f"\n[{region}] 추가 코퍼스 처리 중...")

            for subdir in subdirs:
                dir_path = os.path.join(file_dir, subdir)
                if not os.path.exists(dir_path):
                    print(f"⚠️ 경로 없음: {dir_path}")
                    continue

                json_files = [f for f in os.listdir(dir_path) if f.endswith(".json")]

                for filename in tqdm(json_files, desc=f"{region}-{subdir} 추가 진행도"):
                    file_path = os.path.join(dir_path, filename)
                    try:
                        with open(file_path, "r", encoding="utf-8-sig") as f_in:
                            data = json.load(f_in)

                            text = data.get("transcription", {}).get("dialect", "")

                            if isinstance(text, str) and text.strip():
                                cleaned_text = re.sub(r'\([^)]*\)|\[[^\]]*\]', '', text)
                                cleaned_text = re.sub(r'[^가-힣a-zA-Z0-9.,?! ]', '', cleaned_text)
                                cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

                                if cleaned_text:
                                    output_line = cleaned_text + "\n"
                                    f_out.write(output_line)

                                    line_bytes = len(output_line.encode('utf-8'))
                                    stats[region]["count"] += 1
                                    stats[region]["size"] += line_bytes
                                    total_count += 1
                                    total_size += line_bytes

                    except Exception as e:
                        print(f"\n⚠️ 파일 오류 발생: {filename} - {e}")

    # --- 결과 출력 (논문용 Table 데이터) ---
    print("\n" + "="*50)
    print(f"{'지역':<10} | {'문장 수 (Sentences)':<20} | {'용량 (Size)'}")
    print("-" * 50)
    
    for region, data in stats.items():
        count = data["count"]
        size_mb = data["size"] / (1024 * 1024)
        print(f"{region:<10} | {count:>18,} | {size_mb:>10.2f} MB")
    
    print("-" * 50)
    total_size_gb = total_size / (1024 * 1024 * 1024)
    print(f"{'합계':<10} | {total_count:>18,} | {total_size_gb:>10.2f} GB")
    print("="*50)
    print(f"✅ 최종 코퍼스가 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    prepare_corpus()