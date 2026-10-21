# NFS로 입력을 전달하고 4090 서버에서 실행하기

A6000 서버의 `/data/ohdg/kodialect-bert`를 4090 서버의 같은 경로에서 읽을 수 있도록 NFS가 연결된 구성을 기준으로 한다. NFS 마운트 설정은 사용자 또는 서버 관리자가 진행한다.

**NFS는 코드와 준비된 입력을 전달하는 용도**로 사용한다. 가상환경, 토큰 캐시, 학습 출력은 4090 서버의 로컬 폴더 `/data/ohdg/translation_4090_run`에 만든다. NFS 경로에서 환경 설치나 학습을 직접 실행하지 않는다. 공유 저장소의 `git pull`도 두 서버에서 동시에 실행하지 않는다.

## 1. A6000 서버: 데이터와 MLM 인코더 내보내기

새 도구가 Git에 반영된 상태에서 **A6000 서버**에서 실행한다. 전처리된 데이터와 네 MLM 인코더를 압축 없이 `shared_inputs`에 모은다.

```bash
cd /data/ohdg/kodialect-bert/261021_translation_4gpu

/data/ohdg/anaconda3/envs/dialect_translate/bin/python tools/pack_inputs.py \
  --source-server-root /data/ohdg/kodialect-bert/261012_common_preprocessing_reexperiment/server \
  --output-dir /data/ohdg/kodialect-bert/261021_translation_4gpu/shared_inputs \
  --resolve-tokenizer-code
```

KoBERT에는 가중치·어휘 외에 사용자 정의 토크나이저 코드가 필요하다. `--resolve-tokenizer-code`는 기존 환경의 캐시에서 토크나이저를 불러와 배포용 폴더에 코드와 로컬 참조 설정을 함께 저장하고, 어휘·특수 토큰·예문 토큰 번호가 동일한지 확인한다. **원래 MLM 모델과 토크나이저 파일은 수정하지 않는다.** 캐시에서 코드를 찾을 수 없으면 중단하며 임의의 최신 구현을 다운로드하지 않는다.

이 내보내기에는 준비된 데이터와 인코더 4개가 모두 포함되므로 별도의 데이터 압축파일은 필요 없다. 기존 번역 디코더·학습 결과·캐시는 포함하지 않는다. 이미 존재하는 `shared_inputs`는 덮어쓰지 않으므로 재내보내기가 필요하면 새 폴더 이름을 지정한다.

## 2. 4090 서버: 로컬 코드 작업 폴더 만들기

아래 명령부터는 **4090 서버**에서 실행한다. 먼저 NFS 공유 위치와 로컬 작업 위치의 파일시스템·여유 공간을 확인한다.

```bash
findmnt -T /data/ohdg/kodialect-bert
df -h /data/ohdg
python3 --version

cd /data/ohdg/kodialect-bert/261021_translation_4gpu
python3 tools/create_local_workspace.py \
  --destination /data/ohdg/translation_4090_run
findmnt -T /data/ohdg/translation_4090_run
```

첫 `findmnt` 결과에서 공유 저장소가 `nfs` 또는 `nfs4`로 연결됐는지 확인한다. `python3`는 3.10 이상이어야 한다. 이 도구는 코드만 복사하며 데이터, 모델, 가상환경, 캐시와 실행 결과는 복사하지 않는다. 기존 목적지나 NFS로 확인되는 목적지는 거부한다. 마지막 `findmnt` 결과에서는 실행 폴더가 로컬 파일시스템인지 확인한다. `/data/ohdg` 자체도 NFS라면 로컬 디스크의 다른 경로를 지정한다. 실험 공간은 여유 **200 GiB 이상**을 권장하며 실행기의 기본 시작 검사 기준은 100 GiB다.

## 3. 4090 서버: 환경 설치와 입력 복사

```bash
cd /data/ohdg/translation_4090_run
bash setup_environment.sh --bootstrap-conda
PY="$PWD/.env/bin/python"

"$PY" tools/verify_inputs.py \
  --import-directory /data/ohdg/kodialect-bert/261021_translation_4gpu/shared_inputs \
  --root .
"$PY" tools/verify_inputs.py --root .
```

입력은 NFS에서 로컬 `data/`, `models/`, `provenance/`로 복사되며 SHA256과 데이터 구성 검증을 통과한 후 반영된다. NFS 원본은 변경하지 않는다. 가져오기가 이미 완료됐다면 `--import-directory`를 반복하지 않고 마지막 검증 명령만 실행한다. 새 터미널에서는 로컬 폴더로 이동한 후 `PY="$PWD/.env/bin/python"`을 다시 설정한다.

## 4. GPU 점검 후 학습 시작

```bash
"$PY" tools/smoke_test.py \
  --gpus 0 1 2 3 \
  --require-inputs --inputs-root models/mlm \
  --production-shapes --production-batches

"$PY" run_parallel.py --run-id 4090_fresh_01 --gpus 0 1 2 3 --check
```

스모크테스트와 사전 검사가 성공한 뒤 실행한다.

```bash
nohup "$PY" -u run_parallel.py \
  --run-id 4090_fresh_01 --gpus 0 1 2 3 \
  > launcher.log 2>&1 &
```

기존 MLM 인코더를 사용하여 표준어 CLM 디코더를 새로 학습한 다음, GPU마다 토크나이저 하나의 번역 학습을 5개 시드로 진행한다. `configs/rtx4090.json`의 설정은 그대로 사용한다. 유효 학습 배치는 256이며, 전체 20회 실험이다.

```bash
"$PY" run_parallel.py --run-id 4090_fresh_01 --gpus 0 1 2 3 --status
tail -n 80 launcher.log
"$PY" tools/estimate_runtime.py --run-dir runs/4090_fresh_01
```

결과는 로컬 `runs/4090_fresh_01/results/final_results.md`에 저장된다. 환경·배치·재개 방법의 상세 설명은 [README.md](README.md)를 참고한다.
