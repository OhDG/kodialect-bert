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
    file_dir = "../../project1_dataset"
    output_file = "dialect_corpus.txt"
    
    # 지역별 통계를 저장할 딕셔너리
    stats = {region: {"count": 0, "size": 0} for region in region_dirs}
    total_count = 0
    total_size = 0

    print("--- 대용량 데이터 추출 및 통계 분석 시작 ---")
    
    with open(output_file, "w", encoding="utf-8") as f_out:
        for region, subdir in region_dirs.items():
            dir_path = os.path.join(file_dir, subdir)
            if not os.path.exists(dir_path):
                print(f"⚠️ 경로 없음: {dir_path}")
                continue
            
            print(f"\n[{region}] 처리 중...")
            json_files = [f for f in os.listdir(dir_path) if f.endswith(".json")]
            
            for filename in tqdm(json_files, desc=f"{region} 진행도"):
                file_path = os.path.join(dir_path, filename)
                try:
                    with open(file_path, "r", encoding="utf-8-sig") as f_in:
                        data = json.load(f_in)
                        utterances = data.get("utterance", [])
                        for u in utterances:
                            text = u.get("dialect_form", "")
                            if isinstance(text, str) and text.strip():
                                # 1. 전처리 (괄호 제거)
                                cleaned_text = re.sub(r'\([^)]*\)|\[[^)]*\]', '', text)
                                # 2. 허용된 문자 외 제거
                                cleaned_text = re.sub(r'[^가-힣a-zA-Z0-9.,?! ]', '', cleaned_text).strip()
                                
                                if cleaned_text:
                                    # 파일에 쓰기
                                    output_line = cleaned_text + "\n"
                                    f_out.write(output_line)
                                    
                                    # 통계 집계
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
        # 바이트 단위를 MB로 변환
        size_mb = data["size"] / (1024 * 1024)
        print(f"{region:<10} | {count:>18,} | {size_mb:>10.2f} MB")
    
    print("-" * 50)
    total_size_gb = total_size / (1024 * 1024 * 1024)
    print(f"{'합계':<10} | {total_count:>18,} | {total_size_gb:>10.2f} GB")
    print("="*50)
    print(f"✅ 최종 코퍼스가 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    prepare_corpus()