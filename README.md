# SemanticBrowser

UE5 프로젝트의 에셋을 한국어 문장으로 찾는 의미 검색 PoC입니다. 대상은 LyraRPG `Content/Asian_Village` 아래의 StaticMesh `SM_*` 243개입니다.
모델은 `google/siglip2-base-patch16-224`이고 로컬 CPU에서 돌립니다.

성공 기준과 작업 규칙은 정답표 커밋(`095c745`) 시점의 지시서를 따릅니다. 정답표 `answer_key.csv`는 모델을 실행하기 전에 확정했고, 이후 수정하지 않았습니다.

## 파일

| 파일 | 역할 | 커밋 |
|---|---|---|
| `answer_key.csv` | 정답표 (질의 13건) | O |
| `tools/extract_thumbs.py` | 1단계: UE 에디터 Python으로 썸네일 PNG 추출 | O |
| `tools/check_thumbs.py` | 썸네일을 숫자로 점검 (크기, 배경, 잘림) | O |
| `thumbs/CONDITIONS.md` | 썸네일 촬영 조건과 실제 실행 명령 | O |
| `build_index.py` | 2단계: 이미지 임베딩 색인 `index.npz` 생성 | O |
| `search.py` | 3단계: 문장 검색 (sigmoid 점수) | O |
| `evaluate.py` | 4단계: 채점 → `results/<타임스탬프>/` | O |
| `serve.py` | 에디터 연동용 로컬 검색 서버 (127.0.0.1) | O |
| `ue/sb.py` | 에디터 Output Log에서 검색 → 콘텐츠 브라우저 선택 | O |
| `thumbs/*.png`, `manifest.csv`, `failed.csv`, `index.npz`, `results/**/report.html` | 에셋에서 나온 산출물 | **X** (Fab/Epic 콘텐츠) |

## 준비

```bash
python -m venv --system-site-packages .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

- torch는 CPU 빌드를 씁니다(`requirements.txt` 주석 참고).
- 모델 가중치는 처음 실행할 때 Hugging Face에서 받아 로컬 캐시에 저장합니다.
- 이미지는 외부로 전송하지 않습니다.

## 실행 순서

### 1. 썸네일 추출 (UE 5.8 에디터)

`thumbs/CONDITIONS.md`의 "실행 명령"을 그대로 실행합니다. 에디터가 하나 더 떠서 243개를 찍고, 끝나면 스스로 닫힙니다(약 30분).

- 이미 있는 PNG는 건너뜁니다. 조건을 바꿔 다시 찍으려면 `thumbs/*.png`를 먼저 다른 곳으로 옮기세요.
- `-ExecutePythonScript`는 쓰지 마세요. 스크립트 실행 직후 에디터가 종료됩니다.
- `-ExecCmds` 안에서는 쉼표를 쓸 수 없습니다. 쉼표가 명령 구분자이기 때문입니다.

결과 점검:

```bash
python tools/check_thumbs.py --dir thumbs --out thumbs/_check.csv --tol 12
```

### 2. 색인

```bash
.venv/Scripts/python build_index.py --manifest manifest.csv --out index.npz --model google/siglip2-base-patch16-224 --batch-size 16
```

`index.npz`가 이미 있으면 덮어쓰지 않고 중단합니다.

### 3. 검색

```bash
.venv/Scripts/python search.py --index index.npz --query "등불"
```

- 옵션: `--top-k`(기본 5), `--threshold`(기본 0 = 사용 안 함)
- 점수는 `sigmoid(exp(logit_scale) × 코사인 + logit_bias)`이고, 반올림하지 않고 출력합니다.
- 질의는 SigLIP 2 학습 조건에 맞춰 소문자로 바꾼 뒤, `padding="max_length", max_length=64`로 처리합니다.

### 4. 채점

```bash
.venv/Scripts/python evaluate.py --answer-key answer_key.csv --index index.npz --manifest manifest.csv --results-dir results --top-k 5 --sweep-min 1e-6 --sweep-max 1 --sweep-n 61 --min-pass 8 --min-remaining 8
```

실행할 때마다 `results/<타임스탬프>/` 폴더를 새로 만들고, 아래 파일을 저장합니다.

- `summary.md`: 성공 기준 판정, SigLIP 2 vs 기준선 표, 임계값 구간
- `per_query.csv`: 질의별 상세
- `sweep.csv`: 임계값별 집계
- `report.html`: 썸네일로 눈 검산 (커밋 금지)

## 테스트

```bash
.venv/Scripts/python -m unittest discover -s tests -v
```

- `tests/test_logic.py`: 모델 없이 도는 단위 테스트입니다. 정답표가 확정 커밋(`095c745`) 이후 바뀌지 않았는지, 점수 공식(sigmoid, softmax 아님), 정렬과 동점 처리, 기준선 이름 검색, 채점, 임계값 구간 계산을 검사합니다.
- `tests/test_regression.py`: 커밋된 채점 결과(`results/20261117_171146/per_query.csv`)가 지금 코드로 그대로 재현되는지 검사합니다. `index.npz`가 있을 때만 돌고, 없으면 건너뜁니다.
- `tests/test_ue_editor.py`: 실제 UE 5.8 에디터를 한 번 띄워서 확인하는 통합 테스트입니다(약 1분). 환경 변수 `SB_UE_EDITOR`, `SB_UPROJECT`가 없으면 건너뜁니다.
  - 에디터 검색: `ue/sb.py`를 Output Log 명령으로 실행해, 결과가 `search.py`와 같은지와 콘텐츠 브라우저 선택까지 실행됐는지 봅니다.
  - 썸네일 추출: `CONDITIONS.md`와 같은 조건으로 3개를 다시 찍어, 숫자 조건이 맞는지와 기존 PNG와 픽셀까지 같은지(재현성) 봅니다.

  ```bash
  SB_UE_EDITOR="C:/Program Files/Epic Games/UE_5.8/Engine/Binaries/Win64/UnrealEditor.exe" SB_UPROJECT="C:/Unreal Projects/LyraRPG/LyraRPG.uproject" .venv/Scripts/python -m unittest tests.test_ue_editor -v
  ```

## 에디터에서 쓰기

UE에 들어 있는 Python에는 torch가 없습니다. 그래서 검색은 로컬 서버(`.venv`)가 맡고, 에디터는 질의 문장만 보냅니다. 서버는 127.0.0.1에서만 접속을 받습니다.

1. 검색 서버를 켜 둡니다. 모델을 불러오는 데 몇 초가 걸리고, 이후 질의는 0.1초 안팎입니다.

   ```bash
   .venv/Scripts/python serve.py --index index.npz --manifest manifest.csv --port 8765 --default-k 10
   ```

2. 에디터의 Output Log 입력칸을 **Cmd**로 두고 아래처럼 입력합니다.

   ```
   py "C:/Claude/SemanticBrowser/ue/sb.py" 길가에 세운 가로등
   py "C:/Claude/SemanticBrowser/ue/sb.py" 등불 --k 5
   py "C:/Claude/SemanticBrowser/ue/sb.py" wooden fence --select name
   ```

   - Output Log에 SigLIP 2 순위와 이름 검색 순위를 함께 출력합니다.
   - 콘텐츠 브라우저에서 결과 에셋을 선택해 보여 줍니다. 기본은 SigLIP 결과이고, `--select name`이면 이름 검색 결과, `--select none`이면 선택하지 않습니다.
   - 기본은 임계값 없이 상위 k개를 보여 줍니다. PoC에서 한국어 임계값 구간이 매우 좁았기 때문입니다.

에셋이 추가되면 1단계(썸네일)와 2단계(색인)를 다시 돌린 뒤 서버를 재시작하세요. 이미 있는 썸네일은 건너뛰고, 색인은 새 경로로 만들거나 기존 파일을 옮긴 뒤 만듭니다.

## 결과 해석

> Claude가 쓴 초안을 작성자가 검토함 (2026-11-18). 숫자는 모두 `results/20261117_171146/`에서 가져왔습니다. "추정"이라고 쓴 부분은 확인하지 않은 가설입니다.

### 한 줄 요약

SigLIP 2는 **순위를 매기는 용도**로는 기준선(이름 검색)보다 분명히 낫습니다. 하지만 이 데이터로는 **임계값으로 "없음"을 판정하는 근거가 약합니다.**

### 성공 기준

- **기준 1 (한국어 top-5 통과 8/10 이상): 통과, 9/10.** 영어도 9/10입니다. 둘 다 실패한 질의는 q04 `짚으로 엮은 지붕` / `straw roof`이고, 정답이 각각 52위와 53위였습니다.
- **기준 2 (임계값으로 absent 3건 비우기): 구간 B는 있지만, 운영 임계값으로 쓰기는 어렵습니다.**
  - 한국어 구간 B는 `0.00022876750077403089 < t ≤ 0.00026111684292729885`입니다.
    - 아래 경계는 `피자` 질의의 1위 점수, 위 경계는 q05 `사람 모양 돌 조각`의 정답 점수입니다.
    - 위아래 경계의 차이가 약 1.14배뿐이라, 로그 격자 61개 점 중 1개만 이 구간에 들어갑니다.
  - 이 구간은 같은 13건의 점수를 보고 **나중에 계산한** 범위입니다. 새 질의에서도 맞는다는 근거는 없습니다.
  - 구간 A(정답 10건이 모두 남음)는 한국어와 영어 모두 **없습니다.** q03 `폭포`의 한국어 정답 점수(0.00010723218168113421)가 absent 질의 `피자`의 1위 점수보다 낮기 때문입니다.

### SigLIP 2 vs 기준선 (영어 질의, 같은 채점 기준)

| | SigLIP 2 | 기준선 |
|---|---|---|
| 통과 | 9/10 | 5/10 |
| top-5 안 함정 | 2개 (q08) | 0개 |

- **SigLIP만 통과한 질의: q05, q06, q08, q09.** 이름에 없는 표현(`stone figure of a person`)이나, 이름과 형태가 다른 단어(`arched` ↔ `arch`, `planter` ↔ `flower_box`, `lamp post` ↔ `lamppost`)로 찾는 질의입니다.
- **기준선만 통과한 질의: 없음.**
- 이름에 단어가 그대로 들어 있는 질의(q01, q02, q03, q07, q10)는 둘 다 통과했습니다. 이 경우 첫 정답 순위는 기준선이 같거나 더 앞입니다. 예를 들어 q02 `market stall`은 기준선 1위, SigLIP 2위입니다.
- 함정은 q08 `planter with flowers`에서만 나왔습니다. SigLIP top-5에 `SM_bush_flower_01`, `SM_bush_flower_02`가 들어왔는데, 꽃이 있는 덤불이라 겉모습이 비슷합니다. 기준선의 함정이 0개인 것은 결과 자체가 적었던 영향도 있습니다. q08의 기준선 결과는 1개뿐이었습니다.
- 그래서 실제로 쓴다면 둘 중 하나를 고르기보다 **두 결과를 나란히 보여 주는 구성**이 맞아 보입니다. 에디터 연동(`ue/sb.py`)도 이렇게 만들었습니다.

### 한국어 vs 영어

순위로 보면 한국어도 영어만큼 맞힙니다(통과 9/10 동일). 하지만 **점수의 절대값은 대체로 한국어가 훨씬 낮습니다.**

| 질의 | 한국어 첫 정답 점수 | 영어 첫 정답 점수 |
|---|---|---|
| q01 등불 / lantern | 0.04543487080411636 | 0.5477199995219779 |
| q06 아치형 출입구 / arched gateway | 0.028470966690440202 | 0.5319758211482262 |
| q08 꽃이 심긴 화분 / planter with flowers | 0.0015911564006466723 | 0.2976013569854761 |
| q09 길가에 세운 가로등 / street lamp post | 0.09484444904035258 | 0.6525893920609808 |
| q10 나무 울타리 / wooden fence | 0.496853793275822 | 0.23551921953506602 |

q10만 반대로 한국어 점수가 더 높습니다. 한국어 점수가 낮으면 정답과 absent 질의의 점수 차가 줄어서, 기준 2의 구간이 좁아집니다.

### 실패와 경계 사례

- **q04 `짚으로 엮은 지붕` (실패).** 이미지끼리 비교하면 `SM_thatched_roof_01`~`_04`는 서로 코사인 0.93 이상(쌍별 최소 0.9369043111801147)으로 잘 묶입니다. 그런데 텍스트 질의에서는 50위 밖으로 밀렸습니다. 이미지 표현보다는 "짚 지붕 / straw roof"라는 말과 이 에셋의 그림체를 잇는 쪽이 약한 것으로 **추정**합니다.
- **q03 `폭포`.** 정답 썸네일 `SM_waterfall`은 물 흐름 애니메이션 때문에 그림이 안정되지 않아, 유일하게 300프레임 시점에 저장한 썸네일입니다(`manifest.csv`의 `note`). 한국어 점수가 낮은 것과 관련이 있는지는 확인하지 않았습니다.
- **absent 질의.** 1위 점수 최대값은 한국어 `피자` 0.00022876750077403089, 영어 `computer` 0.0003105472785481782로 작습니다. 하지만 가장 약한 정답 점수와 겹치는 범위에 있습니다.

### 한계

- **질의 수가 적습니다.** 질의가 13건(정답 있음 10건)뿐이라, 한 건이 10%p입니다. 9/10과 8/10의 차이는 질의 하나입니다.
- **범위가 좁습니다.** 에셋 팩 하나(StaticMesh 243개)에서 썸네일 한 각도로만 쟀습니다. 에셋이 늘면 오답 후보도 늘어납니다.
- **임계값은 사후 계산입니다.** 임계값 구간을 채점과 같은 데이터에서 구했기 때문에, 일반화 근거가 되지 못합니다.
- **눈 검산은 아직 기록되지 않았습니다.** `report.html` 검산 결과는 (작성자 확인 후 기입)입니다.

### 다음에 판단하려면 (제안)

- 정답표를 만들 때처럼 모델 실행 전에 질의를 따로 더 만들어(held-out), 이번 구간 B의 임계값이 새 질의에서도 absent를 가르는지 확인합니다.
- 한국어 질의를 영어로 바꿔 검색하는 방식과 비교합니다. 영어 점수가 더 높고 구간 B도 더 넓었으므로 후보가 됩니다. 다만 번역 단계를 추가하는 것은 PoC 범위 밖의 결정입니다.
