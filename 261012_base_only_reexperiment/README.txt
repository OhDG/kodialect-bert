새 기존 발화 자료로 재실험: 로컬 데이터 준비 / 연구실 서버 학습 분리

폴더
- local_prepare: JSON 선택, 동일 전처리, 파일 단위 분할, 두 과제의 입력 파일 생성
- server/region_classification: 260819_4tokenizer_5seed_region_classification 원본 코드
- server/translation: 260811_dialect_to_standard_4tokenizer_seq2seq 원본 코드
- server: 완성된 데이터와 실행 파일을 함께 옮길 서버 전송 폴더

확정한 원천 자료
- 강원도: D:\방언 데이터 총정리\013.한국어 방언 발화 데이터(강원도)의 JSON 전체
- 경상도: D:\방언 데이터 총정리\014.한국어 방언 발화 데이터(경상도)의 JSON 전체
- 전라도: D:\documents\project_1\git\project1_dataset\JSON만_모은폴더_전라도만 사용
- 제주도: D:\방언 데이터 총정리\016.한국어 방언 발화 데이터(제주도)의 JSON 전체
- 충청도: D:\방언 데이터 총정리\017.한국어 방언 발화 데이터(충청도)의 JSON 전체
- 네 지역의 제공 Training/Validation은 원천 자료로 합친 뒤 기존 실험 방식으로 다시 분할한다.
- 중·노년층 자료, ZIP 중복본, JSON 없이 TXT만 있는 자료는 사용하지 않는다.
- 전라도의 새 Training TXT와 새 Validation JSON은 모두 제외한다.

전라도 Validation 판단
기존 전라도와 새 Validation을 전수 비교한 결과 파일 ID 중복은 없지만,
새 275,137발화 중 10,113개(3.6756%)가 기존 방언 문자열과 같았다.
대부분 짧은 표현이며 20자 이상 일치는 32개였다. 동일 녹음의 중복을 뜻하는 것은 아니다.
사용자는 비교 결과를 확인한 뒤 기존 전라도만 사용하는 구성을 선택하였다.
보고서: local_prepare/overlap_audit/jeolla_sentence_overlap.txt
기존 자료 내부의 반복 문장을 새로 제거하는 규칙은 추가하지 않는다.

보존하는 데이터 처리
- 지역 분류: 원본 clean_base_text를 그대로 사용, 빈 결과 제외
- 번역: 원본 normalize_text(NFKC 및 공백 정규화)를 사용, 유효 방언-표준어 쌍 추출
- 지역별 기존 base source_group을 유지하고 파일 ID 순으로 목록을 고정
- 원본 assign_file_splits 함수와 seed42, 목표 비율8:1:1, 파일 전체 배정 방식 유지
- 목표 10%에 도달하거나 넘는 파일까지 배정하므로 실제 비율은 근사치
- 번역은 지역 분류의 파일 분할을 그대로 공유하며 별도로 다시 분할하지 않음
- 원래 제공된 Validation과 새 실험의 validation/test는 서로 다른 개념
- 파일 분할은 동일 파일의 혼입을 막으며, 전체 화자 독립성이나 문장 문자열 무중복을 보장하지 않음

로컬 재현 (새 출력 디렉터리에서만 실행; 기존 생성물 덮어쓰기 금지)
1. local_prepare에서 Python 가상환경 및 NumPy 준비:
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements_local.txt
2. 새 실험 루트에서 실행:
   .\local_prepare\run_local.ps1
3. 전체 입력 행/원본 코드/전송 파일 검증:
   .\local_prepare\.venv\Scripts\python.exe .\server\verify_bundle.py --full
4. 실행 전 확인:
   .\local_prepare\.venv\Scripts\python.exe .\server\run_all.py --check-only

기존 전수 집계 결과를 --audit-counts로 지정하면 첫 문장 수 계산을 재사용할 수 있다.
이 경우 동일 전처리 함수인지 검사하고, 파일 생성 시 모든 원본 JSON의 SHA256과
실제 추출 문장 수를 다시 대조한다. 원본이 달라졌으면 구축을 중단한다.
기본 실행은 원본 JSON을 직접 읽어 새로 집계하므로 이전 감사 파일에 의존하지 않는다.

완료 확인
- local_prepare/build_report.json: 확정 출처, 파일 수, 과제/분할별 입력 수
- server/verification_report.json: 전체 행 검증 결과
- server/bundle_ready.json: 전체 검증 성공 후에만 completed=true
- server/transfer_manifest.json: 옮길 파일의 상대경로, 크기, SHA256
- server/original_code_sha256.json: 원본 학습/평가 코드와 requirements 18개의 해시

서버에는 server 폴더 전체를 전송한다. 로컬에서 학습은 실행하지 않는다.
실행 명령과 기존 환경 사용 방법은 server/SERVER_RUN.txt에 정리했다.
Git은 대용량 준비 데이터를 제외하므로 Git clone/pull만으로 데이터 전송이 완료되지는 않는다.
