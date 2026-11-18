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

(작성 예정)
