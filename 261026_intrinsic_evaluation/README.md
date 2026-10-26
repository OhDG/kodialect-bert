# 새 평가 분할의 토크나이저 내재평가

공통 전처리된 **평가 문장 1,111,735개**와 실제 MLM 학습에 사용한 네 토크나이저를 CPU에서 비교합니다. 기본 실행 위치는 기존 데이터와 MLM 출력이 있는 **A6000 서버**입니다. GPU를 점유하거나 토크나이저를 새로 학습하지 않으며 모델 가중치도 로드하지 않습니다. 평가 결과가 나오기 전에는 이 폴더의 코드나 작은 스모크 결과를 논문 실험 결과로 사용하지 않습니다.

## 서버 실행: 기본 권장 순서

로컬에서 이 폴더를 Git에 반영한 뒤 서버에서 최신 코드를 가져옵니다. 이 도구가 Git commit/push/pull을 자동 수행하지는 않습니다.

코드만 묶은 전송용 압축파일 `artifacts/intrinsic_server_bundle.zip`도 준비되어 있습니다. 이 압축에는 Windows 가상환경, 데이터, 모델 가중치 및 시험 결과가 포함되지 않습니다. Git으로 받는 대신 압축을 올린 경우 서버의 `/data/ohdg/kodialect-bert` 아래에서 해제하면 같은 폴더 구조가 됩니다. 이후에는 아래의 `git pull`을 생략하고 실행 폴더로 이동하면 됩니다.

```bash
cd /data/ohdg/kodialect-bert
git pull
cd 261026_intrinsic_evaluation

bash run_server.sh --setup
bash run_server.sh --check
bash run_server.sh --smoke
```

`--setup`은 프로젝트 **`.env`에만** Python 가상환경과 고정 패키지를 설치합니다. 기본적으로 `/data/ohdg/anaconda3/envs/dialect1/bin/python` 등 기존 Python 3.10 실행 파일을 가상환경 생성의 기반으로만 사용합니다. 기존 `dialect1` 패키지와 시스템 Python·NVIDIA 드라이버를 변경하지 않으며, sudo/apt/conda install 명령을 실행하지 않습니다. 프로젝트 Python이 준비되면 이후 실행은 항상 `.env/bin/python`을 사용합니다.

기반 Python을 직접 지정해야 하면 다음과 같이 실행합니다.

```bash
bash run_server.sh --setup --python /data/ohdg/anaconda3/envs/dialect1/bin/python
```

서버 입력 기본 경로:

```text
원본 서버 루트:
  /data/ohdg/kodialect-bert/261012_common_preprocessing_reexperiment/server
평가 문장:
  region_classification/data/corpus/dialect_test_corpus.txt
원본 토크나이저:
  region_classification/outputs/mlm/{dialect,klue,kobert,mbert}/final_model
```

`--check`와 실제 평가는 먼저 `tools/prepare_server_inputs.py`로 토크나이저 파일만 프로젝트의 `inputs/tokenizers`에 준비합니다. KoBERT의 필요한 커스텀 코드는 저장된 원본이나 기존 캐시/portable export에서 확인합니다. 원본 모델·코퍼스는 읽기만 합니다. 이어 패키지 버전, 평가 코퍼스 존재, 실제 토크나이저 로딩과 짧은 예문 토큰화, 어휘 크기 **32,000 / 32,000 / 8,002 / 119,547**을 검사합니다. `--check`는 빠른 준비 검사이므로 전체 코퍼스를 읽지 않으며, **1,111,735문장 여부와 UTF-8 검사는 전체 평가에서 수행**합니다. 준비 검사가 실패하면 평가를 시작하지 않습니다.

스모크가 성공하면 전체 평가를 실행합니다.

```bash
mkdir -p logs
nohup bash run_server.sh --full > logs/intrinsic_full.log 2>&1 < /dev/null &
tail -F logs/intrinsic_full.log
```

`tail`에서 Ctrl+C를 눌러도 nohup으로 시작한 평가는 계속됩니다. 결과는 `results/full/intrinsic_results.md`와 `results/full/table_03_intrinsic.tex`에서 확인합니다. 완료 시 `[OK] FULL results saved`가 출력되고, 최종 JSON에 `completed: true`, `scope: full`이 기록됩니다.

기본 경로가 다른 서버에서는 모든 실행 모드에 같은 원본 서버 루트를 지정합니다.

```bash
bash run_server.sh --check --source-server-root "/실제 경로/261012_common_preprocessing_reexperiment/server"
bash run_server.sh --full --source-server-root "/실제 경로/261012_common_preprocessing_reexperiment/server" --output-dir results/full_02 --batch-size 1024
```

`--smoke-lines 1000`으로 스모크 문장 수를 지정할 수 있습니다. 경로에 공백이 있으면 큰따옴표로 감쌉니다. 패키지 캐시·커스텀 코드 캐시·결과·로그는 이 프로젝트 폴더 아래에 저장됩니다.

## 로컬 실행: 필요한 경우에만

서버 실행에는 아래 압축·전송 및 Windows 설치 절차가 필요하지 않습니다. 서버에서 실행할 수 없는 경우 같은 지표를 로컬 CPU에서도 계산할 수 있도록 유지한 선택 경로입니다.

## 1. 서버에서 토크나이저 파일만 압축

아래 도구는 Python 표준 라이브러리만 사용합니다. 서버 가상환경에 패키지를 추가할 필요가 없습니다. 원본 MLM 폴더는 읽기만 합니다.

이 신규 코드 폴더를 Git에 반영한 뒤 A6000 서버 저장소에서 `git pull`을 실행하여 도구 파일을 먼저 받아야 합니다. 기존 `shared_inputs`를 그대로 사용하므로 모델이나 토크나이저를 다시 학습하지 않습니다.

```bash
python3 /data/ohdg/kodialect-bert/261026_intrinsic_evaluation/tools/pack_tokenizers.py \
  --source-root /data/ohdg/kodialect-bert/261021_translation_4gpu/shared_inputs/models/mlm \
  --output /data/ohdg/intrinsic_tokenizers.tar.gz
```

이 경로는 기존에 생성한 portable `shared_inputs`입니다. `{dialect,klue,kobert,mbert}/final_model`의 저장된 `tokenizer_config.json`, 어휘·SentencePiece 모델·special token 설정과 필요한 커스텀 Python 코드를 명시적인 허용 목록으로 가져옵니다. `config.json`은 토크나이저 클래스 해석을 위한 메타데이터이며, `.safetensors`, `.bin`, optimizer 등 가중치는 포함하지 않습니다.

`intrinsic_tokenizers.tar.gz`와 `.sha256`을 로컬로 가져옵니다. KoBERT의 `tokenization_kobert.py`가 없거나 설정이 원격 커스텀 코드만 참조하면 도구가 중단합니다. 어휘를 새로 만들거나 다른 배포본으로 대체하지 마세요.

A6000 서버에서 압축한 경우 로컬 PowerShell에서 다음과 같이 받을 수 있습니다.

```powershell
cd D:\documents\project_1\git\kodialect-bert\261026_intrinsic_evaluation
scp -P 2222 ohdg@203.253.21.179:/data/ohdg/intrinsic_tokenizers.tar.gz .
scp -P 2222 ohdg@203.253.21.179:/data/ohdg/intrinsic_tokenizers.tar.gz.sha256 .
```

## 2. 로컬 전용 가상환경 설치

PowerShell에서 실행합니다. Python 3.10이 필요합니다.

```powershell
cd D:\documents\project_1\git\kodialect-bert\261026_intrinsic_evaluation
.\run_local.ps1 -SetupOnly
```

설치 위치는 이 폴더의 `.env`이며 pip·Hugging Face 캐시는 `.cache`입니다. 시스템 Python, 기존 conda 환경, 서버 NVIDIA 드라이버는 수정하지 않습니다. `run_local.ps1`이 설정하는 환경변수는 실행 후 원래 값으로 돌립니다. PowerShell 실행 정책 때문에 파일 실행이 제한되면 해당 세션에서만 허용하거나, 아래와 같이 직접 프로젝트 Python을 사용해도 됩니다.

```powershell
py -3.10 -m venv .env
.\.env\Scripts\python.exe -m pip install -r requirements.txt
```

이 프로젝트는 CUDA/PyTorch를 요구하지 않습니다. Transformers가 PyTorch 부재를 알리는 문구를 출력하더라도 토크나이저 평가는 CPU에서 가능합니다.

## 3. 압축 가져오기 및 검증

다운로드한 파일 경로를 지정합니다. 기존 폴더를 덮어쓰지 않습니다.

```powershell
.\.env\Scripts\python.exe tools\pack_tokenizers.py --import D:\다운로드\intrinsic_tokenizers.tar.gz
.\.env\Scripts\python.exe tools\pack_tokenizers.py --verify .\inputs\tokenizers
```

압축 경로 탈출, 심볼릭 링크, 중복 파일, 누락 파일, 체크섬 변경을 검사합니다. Hugging Face 다운로드 없이 `local_files_only=True`로 저장된 토크나이저를 로드하며, KoBERT의 커스텀 코드는 가져온 신뢰할 수 있는 원본 코드만 사용합니다.

## 4. 스모크 확인 후 전체 평가

실제 저장된 네 토크나이저로 첫 1,000문장만 확인합니다. 이 결과에는 `scope=smoke`가 표시됩니다.

```powershell
.\run_local.ps1 -MaxLines 1000
```

기본 전체 평가는 다음 명령입니다.

```powershell
.\run_local.ps1
```

기본 입력은 다음 기존 파일이며 새 원천 JSON을 읽거나 다시 전처리하지 않습니다.

```text
..\261012_common_preprocessing_reexperiment\server\region_classification\data\corpus\dialect_test_corpus.txt
```

로컬 입력을 사전 검사한 결과는 1,111,735문장, 공백 기준 7,734,968어절, 공백 제외 20,253,468문자입니다. 파일 SHA256은 `cfc56934f83b8f1e3443aa3c6c9a9a24fa101466fedfa91bad99e30239399e5c`입니다. 이 값은 입력 코퍼스 검사 결과이며, 토크나이저 평가 지표는 별도로 실행해야 산출됩니다.

전체 실행은 비어 있지 않은 문장 수가 **1,111,735개**인지 확인하며, 로드한 어휘 크기가 각각 **32,000 / 32,000 / 8,002 / 119,547**인지도 확인합니다. 작은 시험용 어휘를 전체 결과로 잘못 사용하는 것을 방지합니다. 모든 문장을 처리하며 임의 표집·길이 잘림·패딩·`[CLS]`/`[SEP]` 추가를 적용하지 않습니다. 이전 스모크/결과 폴더가 있으면 덮어쓰지 않으므로 다시 실행할 때는 `-OutputDir .\results\full_02`처럼 새 경로를 지정하세요.

직접 실행 시 지원 인수:

```powershell
.\.env\Scripts\python.exe evaluate.py --corpus "평가코퍼스경로" --tokenizers-root .\inputs\tokenizers --output-dir .\results\full_02 --batch-size 1024 --progress-every 100000
```

토크나이저마다 문장 처리와 CWR 고유 어절 처리 진행 상황을 출력합니다. GPU 학습은 없지만 SentencePiece 기반 KoBERT 및 어절 단위 추가 분석에는 시간이 걸릴 수 있습니다.

## 지표 정의와 기존 코드 보완점

- 기반 코드는 `260630_test_1/step3_test_tokenizer_final.py`입니다. 이 코드는 세 토크나이저만 비교하므로 **KLUE-BERT를 추가**했습니다.
- 기존 코드는 CWR을 계산하지 않습니다. `260331_test_1/step3_test_tokenizer.py`의 **공백 기준 어절을 개별 토큰화하여 2개 이상으로 나뉜 비율**을 추가했습니다. 같은 어절은 한 번만 토큰화하고 출현 횟수로 가중하므로 모든 출현을 개별 계산한 값과 같습니다. 0개 토큰이 되는 어절은 CWR 분모에서 제외하고 별도 집계합니다. UNK 한 개로 처리된 어절은 분절된 어절로 계산되지 않으므로 UNK 비율과 함께 해석합니다.
- Fertility는 기존 정의인 **문장 전체의 토큰 수 / 공백 기준 어절 수**를 유지합니다. 어절을 따로 토큰화한 CWR과 계산 단위가 다를 수 있습니다.
- 공통 전처리된 문장에는 NFKC 변환이나 문자·괄호 삭제를 다시 적용하지 않습니다. UTF-8 오류를 무시하지 않으며, 줄 끝과 앞뒤 공백을 제거하고 빈 줄만 제외합니다.
- UNK는 표시 문자열 대신 각 토크나이저의 `unk_token_id`로 집계합니다.
- 공백 포함/제외 문자 수와 비율을 모두 저장합니다. **논문용 Chars/Token 및 Tokens/Char는 ASCII 공백 제외**입니다. 이는 기존 논문 수치의 문자 수 기준과 일치하며, `260630` 코드의 공백 포함 `len(text)`와 차이가 있으므로 결과 파일에 명시합니다. 준비 코퍼스의 공백은 이미 ASCII 한 칸으로 정규화되어 있습니다.
- Zipf 계산은 `260630`과 같이 자연로그 `log(rank) <= 6`(최대 상위 403개), Simpson 적분, 로그 빈도 선형 회귀를 사용합니다. **Power-law Error는 회귀 잔차의 MAE**입니다. `260331`의 RMSE·사다리꼴 적분과 혼용하지 않습니다.
- Cardinality와 Zipf 지표는 어휘 활용 분포의 보조 정보입니다. 단일 수치만으로 과제 성능의 우열을 단정하지 않습니다.

## 생성 파일

전체 결과는 `results/full/`, 제한 실행은 `results/smoke_1000/` 등에 저장됩니다.

- `tokenizer_comparison_full_results.json`: 반올림하지 않은 모든 지표와 완료·평가 범위 표시
- `tokenizer_comparison_metrics.csv`: 반올림하지 않은 지표 비교표
- `intrinsic_results.md`: 최대 소수점 네 자리의 읽기용 결과
- `table_03_intrinsic.tex`: 논문에 넣을 수 있는 표 소스, 토크나이저 명칭 명시
- `provenance.json`: 평가 코퍼스 원본 SHA256·문장/어절/문자 수, 토크나이저 파일 SHA256, 패키지 버전과 지표 정의
- 토크나이저별 `*_metrics.json`, `*_zipf_curve.csv`

표의 `0.0000%`는 반올림 결과일 수 있으므로 UNK 개수와 원본 JSON을 함께 확인합니다. 이 평가 실행 자체가 논문을 자동 덮어쓰지는 않습니다.

## 검증

```powershell
.\.env\Scripts\python.exe -m unittest discover -s tests -v
.\.env\Scripts\python.exe tools\smoke_test.py
```

단위 테스트는 Fertility·CWR의 빈도 가중, UNK·문자 수, Zipf의 정확한 역순위 분포와 상위 403개 구간, UTF-8 오류와 압축 보안 검사를 확인합니다. 작은 CPU 스모크는 새로 만든 WordPiece **시험용 토크나이저** 네 개로 가져오기·지표·모든 출력의 연결을 검증합니다. 이 스모크는 생산 KoBERT 커스텀 코드나 실제 코퍼스 결과를 대신하지 않습니다.

