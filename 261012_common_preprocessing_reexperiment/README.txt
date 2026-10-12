공통 전처리로 재구축한 다섯 지역 방언 실험

local_prepare/: 로컬 JSON 전처리, 기존 분할 재사용, 집계/검증 테스트와 데이터 포장
server/: 연구실 서버의 토크나이저 학습 이후 전체 실험과 격리된 스모크테스트
artifacts/: Git과 별도로 서버로 옮길 prepared_data.tar.gz와 SHA256

원천 자료
- 강원도/경상도/제주도/충청도: 새 한국어 방언 발화 JSON 자료
- 전라도: 기존 project1_dataset의 JSON만 사용
- 중노년층 추가 자료, TXT-only 자료, 새 전라도 Validation은 포함하지 않음
- 총 34,213개 JSON의 기존 파일별 분할과 순서를 유지. 재분할 및 문장 중복 제거는 하지 않음

이번 변경
- 명확한 (방언)/(표준어)는 방언 입력에 왼쪽, 표준어 정답에 오른쪽을 남긴다.
- 불확실한 이중 괄호 전사 및 해석이 모호한 대응쌍은 해당 발화 전체를 모든 과제에서 제외한다.
- 단독 괄호와 근거가 확정되지 않은 기호는 삭제하지 않고 점검 플래그를 기록한다.
- NFKC/공백 정규화와 위 규칙을 공통 적용하며, 이전의 문자 허용목록 삭제는 사용하지 않는다.
- 같은 발화의 공통 방언 입력을 어휘 학습, MLM, 지역 분류, 번역에 동일하게 사용한다.
- 정책의 근거와 적용 한계: local_prepare/PREPROCESSING_POLICY.txt
- 원본 실험 학습 파일 18개와 학습 하이퍼파라미터는 유지한다.

주요 결과 기록
- local_prepare/build_report.json: 출처/정책 해시, 고정 분할, 구축 규모
- local_prepare/preprocessing_statistics.json: 전체/지역/분할별 처리 및 제외 통계
- local_prepare/preprocessing_per_file.jsonl: 파일별 처리 통계
- local_prepare/source_manifest.csv: 기존 문장 수와 새 문장 수, 고정 파일 분할
- server/verification_report.json: 전체 행/공통 입력/원본 코드/전송 SHA256 검증
- server/smoke_runs/<실행ID>/smoke_report.json: 실제 축소 실행 결과

서버에서는 server/SERVER_RUN.txt 순서로 진행한다.
최종 데이터 규모는 DATA_SUMMARY.txt, 실제 로컬 검증 결과와 범위는 TEST_RESULTS.txt 참고.
지역 분류는 dialect1, 번역은 dialect_translate 환경을 유지한다.
두 환경은 새 MLM 인코더 파일을 공유하며, 가상환경을 합치지 않는다.
Git에는 코드만 포함하므로 artifacts의 준비 데이터 압축과 체크섬도 별도 전송해야 한다.
로컬 가상환경과 기존 실험의 모델/토크나이저/캐시를 서버 새 실험에 복사하지 않는다.
스모크테스트의 합성 데이터, 2-step 학습 및 모델은 연구 결과나 본 학습에 재사용하지 않는다.

로컬 재구축 명령(이미 구축된 출력은 덮어쓰지 않음)
  local_prepare/run_local.ps1 -Workers 8
  python -m unittest discover -s local_prepare -p "test_*.py"
  python server/verify_bundle.py --full
  python local_prepare/pack_data.py

기존 분할의 원본 목록은 local_prepare/frozen_split_manifest.csv이며,
보관된 이전 실험의 source_manifest.csv와 같은 바이트로 복사했다.
서버에는 source JSON을 열지 않고 비교할 수 있는 frozen_split_assignments.json을 제공한다.
