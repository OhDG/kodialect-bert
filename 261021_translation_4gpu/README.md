# RTX 4090 4장 번역 실험

기존 A6000 서버에서 학습한 **MLM 인코더 4개를 그대로 가져와**, 새 서버에서 표준어 CLM 디코더 1개와 번역 모델 **4토크나이저 × 5시드 = 20회**를 새로 학습한다. 지역 분류와 MLM 학습은 실행하지 않는다. 데이터는 기존 공통 전처리와 파일 단위 분할을 유지한다.

디코더를 GPU 0에서 먼저 학습한 다음 아래 작업을 동시에 실행한다. 각 GPU는 맡은 토크나이저의 시드 `13, 21, 42, 87, 100`을 순서대로 처리한다. DDP를 쓰지 않는 독립 프로세스 4개다.

| GPU | 번역 토크나이저 | 실행 수 |
| --- | --- | ---: |
| 0 | dialect | 5 |
| 1 | klue | 5 |
| 2 | kobert | 5 |
| 3 | mbert | 5 |

[NVIDIA RTX 4090 사양](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)은 카드당 24 GB이며 NVLink를 지원하지 않는다. 각 작업에 쓸 수 있는 VRAM은 **24 GB**이고, 4장을 한 모델용 96 GB로 합치는 구성은 아니다.

## 1. 저장 공간과 전달 파일

새 서버의 확인된 사양은 Xeon Silver 4314 2개, 총 32코어/64스레드, RAM 약 251 GiB(확인 당시 가용 약 244 GiB)다. 이 번들은 동시 작업을 위해 RAM 96 GiB 이상을 권장한다. 실제 사용량은 첫 실행에서 확인한다.

작업 공간은 `/data/ohdg`로 두고 **먼저 실제 여유 공간을 확인**한다. `/data`의 용량은 아직 확인되지 않았다. 환경, 압축파일과 해제본, 토큰 캐시, 20회 체크포인트와 예측 결과를 고려해 **여유 공간 200 GiB 이상**을 권장한다. 여유 96 GiB인 홈 디렉터리는 이 실험의 작업 공간으로 잡지 않는다.

실행기의 여유 공간 기준은 새 실행·`--check`에서 100 GiB, `--resume`에서 20 GiB다. `--min-free-gib`로 기준을 지정할 수 있지만, 이 숫자는 시작을 허용하는 최소 기준이며 최종 사용량의 상한이나 공간 부족 방지 보증은 아니다.

```bash
mkdir -p /data/ohdg
df -h /data /data/ohdg
nvidia-smi
```

Git 반영과 서버 간 파일 전송은 사용자가 진행한다. 아래 명령은 기존 저장소 복제본이 `/data/ohdg/kodialect-bert`에 있다는 기준이다. 복제 위치가 다르면 `cd` 경로만 맞춘다.

전달할 파일은 다음과 같다.

- 새 코드 폴더 `261021_translation_4gpu/`.
- 로컬에서 만든 `artifacts/translation_data.tar.gz`와 같은 이름 뒤에 `.sha256`을 붙인 파일. 압축 전 데이터는 약 2.8 GiB다.
- 기존 A6000 서버에서 아래 명령으로 만든 `translation_mlm_models.tar.gz`와 `.sha256` 파일. 로컬 데이터 압축파일에는 MLM 가중치가 들어 있지 않다.

기존 A6000 서버에도 새 코드 폴더를 반영한 뒤 실행한다. 아래 Python 경로는 기존 번역 환경의 경로다.

```bash
cd /data/ohdg/kodialect-bert/261021_translation_4gpu
/data/ohdg/anaconda3/envs/dialect_translate/bin/python tools/pack_inputs.py \
  --models-only \
  --source-server-root /data/ohdg/kodialect-bert/261012_common_preprocessing_reexperiment/server \
  --output /data/ohdg/translation_mlm_models.tar.gz
```

로컬 데이터 압축파일과 서버에서 만든 모델 압축파일, 각각의 체크섬 파일을 모두 새 서버의 `/data/ohdg/`로 옮긴다. 완성된 `translation_data.tar.gz`는 702,620,430바이트(약 703 MB)다. 4개 `final_model` 폴더의 가중치, 설정, 토크나이저 파일을 함께 묶으므로 일부 파일만 골라 복사하지 않는다. 기존 번역 결과, 디코더, 체크포인트와 토큰 캐시는 가져오지 않는다.

## 2. 새 서버 환경 설치

Linux x86_64와 동작하는 NVIDIA 드라이버가 필요하다. 새 폴더에서 실행한다.

```bash
cd /data/ohdg/kodialect-bert/261021_translation_4gpu
bash setup_environment.sh --bootstrap-conda
PY="$PWD/.env/bin/python"
"$PY" --version
```

기존 conda가 있으면 프로젝트 내부 `.env`를 만들고, 없으면 체크섬을 검증한 Miniforge를 `.miniforge`에 설치한다. 기존 conda base와 시스템 드라이버를 수정하지 않는다. 설치에는 인터넷 연결이 필요하다. 전체 설치 버전은 `environment.freeze.txt`에 기록된다.

Python 3.10, PyTorch `2.0.1+cu118`, Transformers `4.40.1`, Accelerate `0.30.1` 등을 고정한다. CUDA 11.8 wheel 설치 경로는 [PyTorch 공식 이전 버전 설치 안내](https://pytorch.org/get-started/previous-versions/#v201)를 따른다. 나머지는 [requirements.txt](requirements.txt)를 참고한다.

이미 Python 3.10이 있고 venv를 쓰려면 위 설치 명령 대신 다음을 사용한다.

```bash
bash setup_environment.sh --venv --python /실제/Python3.10/경로
PY="$PWD/.venv/bin/python"
```

새 터미널에 접속할 때는 해당 폴더에서 `PY="$PWD/.env/bin/python"`을 다시 지정한다. venv를 선택했다면 `.venv`를 쓴다.

## 3. 데이터·모델 가져오기와 GPU 확인

새 서버에서 두 체크섬을 확인하고 각 압축파일을 한 번씩 가져온다.

```bash
(cd /data/ohdg && sha256sum -c translation_data.tar.gz.sha256)
(cd /data/ohdg && sha256sum -c translation_mlm_models.tar.gz.sha256)

"$PY" tools/verify_inputs.py \
  --extract /data/ohdg/translation_data.tar.gz --root . --data-only
"$PY" tools/verify_inputs.py \
  --extract /data/ohdg/translation_mlm_models.tar.gz --root . --models-only
"$PY" tools/verify_inputs.py --root .
```

가져오기는 해시와 파일 목록을 검사하며 기존 파일을 덮어쓰지 않는다. 이미 가져오기가 끝났다면 `--extract`를 반복하지 말고 마지막 검증 명령만 실행한다. 준비된 입력은 `data/`, `models/mlm/`, `provenance/`에 배치된다.

```bash
"$PY" tools/smoke_test.py \
  --gpus 0 1 2 3 \
  --require-inputs --inputs-root models/mlm \
  --production-shapes --production-batches
```

4개 GPU 모두 `PASS`인지 확인한다. 결과와 로그는 `smoke/<실행시각>/`에 남는다. 실제 인코더와 디코더 구조로 학습·저장·재개·생성을 짧게 실행하며, `--production-batches`는 길이 128에서 본 실험의 학습·평가 배치와 beam 4 생성도 검사해 몇 분 이상 걸릴 수 있다. 이 옵션을 빼면 작은 배치 기능 검사만 수행한다. **통과해도 전체 데이터 실행의 속도와 모든 메모리 상황까지 보증하지는 않는다.**

## 4. 계획 확인 후 실제 실행

```bash
"$PY" run_parallel.py --run-id 4090_fresh_01 --gpus 0 1 2 3 --plan
"$PY" run_parallel.py --run-id 4090_fresh_01 --gpus 0 1 2 3 --check
```

`--plan`은 실행 구성을 출력하고 `--check`는 입력·환경·자원을 점검한다. 둘 다 본 학습을 시작하지 않는다. 이 옵션 없이 실행하면 실제 학습이 시작된다.

```bash
nohup "$PY" -u run_parallel.py \
  --run-id 4090_fresh_01 --gpus 0 1 2 3 \
  > launcher.log 2>&1 &
```

[configs/rtx4090.json](configs/rtx4090.json)의 기본 배치는 다음과 같다.

| 단계 | GPU당 학습 배치 | 누적 횟수 | 유효 배치 | 평가 배치 |
| --- | ---: | ---: | ---: | ---: |
| 공유 CLM 디코더 | 64 | 4 | 256 | 128 |
| 번역 20회 | 64 | 4 | 256 | 32 |

각 GPU 작업은 CPU 스레드 4개, 전처리 프로세스 4개, DataLoader worker 2개를 사용한다. 캐시는 새 실행 폴더 안에 만들고 토크나이저별로 나눈다. 같은 토크나이저의 5개 시드는 순차 실행하므로 동일 캐시를 동시에 생성하지 않는다.

기존 유효 학습 배치 256, 3 epoch, 길이 128, 생성 beam 4, 검증 chrF++ 기반 체크포인트 선택은 유지한다. 번역 검증 생성은 고정된 50,000건, 최종 테스트는 1,111,735건이다. 하드웨어, 마이크로배치와 Accelerate 버전 차이가 있으므로 기존 실행과 비트 단위로 같은 결과를 보장하지 않는다. 비교에 사용할 실행 설정과 환경 기록을 함께 보관한다.

## 5. 진행 확인·재개·결과

```bash
"$PY" run_parallel.py --run-id 4090_fresh_01 --gpus 0 1 2 3 --status
tail -n 80 launcher.log
"$PY" tools/estimate_runtime.py --run-dir runs/4090_fresh_01
```

단계별 로그는 `runs/4090_fresh_01/logs/`, 상태는 `runs/4090_fresh_01/state.json`, 실행 설정은 `runs/4090_fresh_01/run_config.json`에 남는다. 완료 후 요약은 **`runs/4090_fresh_01/results/final_results.md`**에서 확인한다.

중단 후에는 같은 run ID와 GPU 인자로 재개한다.

```bash
nohup "$PY" -u run_parallel.py \
  --run-id 4090_fresh_01 --gpus 0 1 2 3 --resume \
  > launcher_resume.log 2>&1 &
```

실행 잠금은 이 번들에서 실행기 하나만 동작하도록 한다. run ID가 달라도 동시에 두 실행기를 띄울 수 없으며, 잠금 파일을 임의로 지우지 않는다. 완료된 작업은 건너뛰며, 미완료 작업은 사용 가능한 체크포인트를 기준으로 재개한다. 재개 시에는 코드·입력·실행 설정도 기존 기록과 같아야 한다. 학습 종료 후 최종 테스트 도중 중단됐다면 평가가 다시 실행될 수 있다. 모델 폴더가 생겼다는 사실만으로 전체 작업이 완료된 것은 아니다.

첫 체크포인트 저장 전 중단 등으로 미완료 출력에 정상적인 모델·optimizer·RNG 체크포인트가 없으면 재개를 거부한다. 이 경우 기존 폴더를 보존하고 새 run ID로 처음부터 실행한다. 임의로 이전 출력 일부를 지워 재개 조건을 우회하지 않는다.

실행 단계 전체 시간은 `results/pipeline_stage_times.json`과 최종 보고서 마지막 표에 따로 기록한다. 재개된 작업의 Trainer 학습·평가 시간은 마지막 호출 구간의 값이므로 중단 없는 실행의 시간과 그대로 비교하지 않는다. 강제 종료로 종료 기록이 없으면 시간 측정이 불완전하다고 표시한다.

## 6. 예상 시간과 검증 범위

경합이 없었던 A6000 로그에서 한 번역 작업은 학습·epoch 검증 약 **5시간 29분**, 마지막 검증 약 **16분**이었다. 테스트 `6,619/17,371` 배치가 `1시간 45분 06초`에 진행된 기록을 비례 계산하면 테스트 전체 약 **4.60시간**, 합계 약 **10.34시간**이다. 초기화·지표 계산·파일 저장을 더해 **1회 10.5–12시간**을 출발점으로 잡는다.

따라서 4개 GPU에 각 5회씩 배정하면 **번역만 약 52.5–60시간**, 여기에 새 디코더 학습과 캐시 준비 시간이 더해진다. 4090의 실제 가속률은 아직 측정하지 않았다. CPU·디스크 경합이 낮고 배치가 정상 실행된다는 조건에서 전체 **대략 2–4일**을 계획 범위로 볼 수 있지만 완료 기한은 아니다. 토크나이저별 속도 차이로 가장 늦은 GPU가 전체 완료 시점을 결정한다.

```bash
# 실행 전: 기존 서버 기준 추정만 출력
"$PY" tools/estimate_runtime.py
# 실행 중: 완료된 작업의 실제 시간으로 보정
"$PY" tools/estimate_runtime.py --run-dir runs/4090_fresh_01
# 디코더의 남은 시간을 별도로 알고 있을 때만 추가
"$PY" tools/estimate_runtime.py --run-dir runs/4090_fresh_01 --decoder-hours 3
```

`--decoder-hours`는 디코더의 **남은 시간**을 직접 더하는 옵션이다. 기본값은 디코더 시간을 모르는 상태로 제외하며 MLM 학습 시간은 계산하지 않는다. 오래 실행 중인 작업과 실패 후 재개는 추가 시간이 불명확하다는 경고를 표시한다. 필요하면 `--json`으로 추정값을 출력할 수 있다.

로컬 검사만으로 새 서버의 4×4090 학습 성공과 속도를 확인할 수 없다. 위 GPU smoke와 실제 첫 단계의 메모리·속도 확인은 새 서버에서 해야 한다. 이 번들 준비 과정에서 기존 A6000 작업 중단이나 원격 업로드는 수행하지 않았다.
