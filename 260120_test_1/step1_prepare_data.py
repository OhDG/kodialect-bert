# import os
# import json
# import re

# def prepare_corpus():
#     region_dirs = {
#         "강원도": "JSON만_모은폴더_강원도",
#         "경상도": "JSON만_모은폴더_경상도",
#         "전라도": "JSON만_모은폴더_전라도",
#         "제주도": "JSON만_모은폴더_제주도",
#         "충청도": "JSON만_모은폴더_충청도"
#     }
#     file_dir = "../../project1_dataset"
#     output_file = "dialect_corpus.txt"
    
#     all_texts = []

#     print("--- 데이터 추출 시작 ---")
#     for region, subdir in region_dirs.items():
#         dir_path = os.path.join(file_dir, subdir)
#         if not os.path.exists(dir_path):
#             print(f"⚠️ 경로 없음: {dir_path}")
#             continue
            
#         count = 0
#         for filename in os.listdir(dir_path):
#             if filename.endswith(".json"):
#                 file_path = os.path.join(dir_path, filename)
#                 try:
#                     with open(file_path, "r", encoding="utf-8-sig") as f:
#                         data = json.load(f)
#                         utterances = data.get("utterance", [])
#                         for u in utterances:
#                             text = u.get("dialect_form", "")
#                             if isinstance(text, str) and text.strip():
#                                 # 특수문자 및 괄호 내용 제거
#                                 cleaned_text = re.sub(r'\([^)]*\)|\[[^)]*\]', '', text)
#                                 cleaned_text = re.sub(r'[^가-힣a-zA-Z0-9.,?! ]', '', cleaned_text).strip()
#                                 if cleaned_text:
#                                     all_texts.append(cleaned_text)
#                 except Exception as e:
#                     print(f"⚠️ 오류 발생: {file_path} - {e}")
                
#                 count += 1
#                 # 테스트를 위해 각 지역별 파일 1개만 읽기 (전체 학습 시 아래 break 제거)
#                 print(f" -> {region} {count}개 파일 로딩 완료.")
#                 # break 

#     # 텍스트 파일로 저장 (토크나이저 학습용 원재료)
#     with open(output_file, "w", encoding="utf-8") as f:
#         for line in all_texts:
#             f.write(line + "\n")
    
#     print(f"--- 총 {len(all_texts)}개의 문장을 {output_file}에 저장했습니다. ---")

# if __name__ == "__main__":
#     prepare_corpus()

import os
import json
import re
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
    
    total_count = 0

    print("--- 대용량 데이터 추출 시작 ---")
    
    # 파일을 쓰기 모드로 미리 엽니다.
    with open(output_file, "w", encoding="utf-8") as f_out:
        for region, subdir in region_dirs.items():
            dir_path = os.path.join(file_dir, subdir)
            if not os.path.exists(dir_path):
                print(f"⚠️ 경로 없음: {dir_path}")
                continue
            
            print(f"\n[{region}] 처리 중...")
            json_files = [f for f in os.listdir(dir_path) if f.endswith(".json")]
            
            # 진행률 표시를 위해 tqdm 사용
            for filename in tqdm(json_files, desc=f"{region} 진행도"):
                file_path = os.path.join(dir_path, filename)
                try:
                    with open(file_path, "r", encoding="utf-8-sig") as f_in:
                        data = json.load(f_in)
                        utterances = data.get("utterance", [])
                        for u in utterances:
                            text = u.get("dialect_form", "")
                            if isinstance(text, str) and text.strip():
                                # 1. 괄호와 그 안의 내용 제거 (주석 등)
                                cleaned_text = re.sub(r'\([^)]*\)|\[[^)]*\]', '', text)
                                # 2. 허용된 문자 외 제거 (한글, 영문, 숫자, 기본 문장부호)
                                cleaned_text = re.sub(r'[^가-힣a-zA-Z0-9.,?! ]', '', cleaned_text).strip()
                                
                                if cleaned_text:
                                    # 리스트에 담지 않고 바로 파일에 기록
                                    f_out.write(cleaned_text + "\n")
                                    total_count += 1
                except Exception as e:
                    print(f"\n⚠️ 파일 오류 발생: {filename} - {e}")
                
                # break  

    print(f"\n--- 처리 완료! ---")
    print(f"✅ 총 {total_count}개의 문장이 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    prepare_corpus()