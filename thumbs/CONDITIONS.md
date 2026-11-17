# 썸네일 촬영 조건

243개 전부 아래의 **같은 조건**으로 찍었습니다. 스크립트는 `tools/extract_thumbs.py`이고, 숫자는 모두 명령줄 인자로 넘겼습니다(아래 "실행 명령" 참고).

## 방법
- UE 5.8 에디터 Python에서 SceneCapture2D로 직접 렌더링했습니다(B안). 에디터 썸네일 캐시는 쓰지 않았습니다.
- 빈 임시 레벨(`/Temp/SB_ThumbLevel_<시각>`, 저장 안 함)에 메시를 하나씩 원점에 놓고 찍었습니다.
- 대상: `/Game/Asian_Village` 아래, 이름이 `SM_`으로 시작하는 StaticMesh. 이름순으로 처리했습니다.

## 이미지
| 항목 | 값 |
|---|---|
| 해상도 | 512 × 512 px, PNG (RGBA8 sRGB, 알파는 전부 255) |
| 캡처 소스 | Final Color (LDR), 톤매핑까지 적용된 색 |
| 배경 | 언릿(Unlit) 단색 구. 발광 선형값 0.22, **PNG에서 측정한 값 RGB = (130, 130, 130) (243장 모두 같음)** |

## 카메라
| 항목 | 값 |
|---|---|
| 방향 | 피치 -11.25°, 요 -157.5°, 롤 0° (UE 기본 StaticMesh 썸네일 각도) |
| 화각 | 30° (원근 투영, 정사각형) |
| 조준 | 메시 바운딩 박스 중심 |
| 거리 | 바운딩 박스 꼭짓점 8개가 모두 화면 안에 들어오는 최소 거리 × 1.05 |
| LOD | LOD0 고정 |

## 조명·후처리
| 항목 | 값 |
|---|---|
| 디렉셔널 라이트 | 2 lux, 피치 -45°, 요 -120°, 그림자 있음 |
| 스카이 라이트 | 회색 배경을 캡처한 Captured Scene, 세기 1.0 |
| 노출 | 수동(Manual), 보정 0 EV, 물리 카메라 노출 사용 안 함 |
| 끈 것 | Lumen GI·반사, Temporal AA, 모션 블러, 블룸, 렌즈 플레어, 안개·대기, 그레인, 비네팅, 눈 적응 |
| 텍스처 스트리밍 | 촬영 중 끔(`r.TextureStreaming 0`) → 모든 텍스처를 최고 해상도로 사용 |
| World Position Offset | 끔 → 나무·대나무의 바람 흔들림 없이 멈춘 모습 |

## 안정화 대기
메시를 바꾼 뒤 최소 10프레임을 기다리고, 그다음 5프레임마다 찍어 직전 그림과 픽셀 단위로 비교했습니다. 3번 연속 같으면 저장했습니다.
300프레임 안에 안정되지 않으면 그 시점의 그림을 저장하고 `manifest.csv`의 `note`에 `unstable_at_max_frames`로 표시했습니다. 물 흐름처럼 시간에 따라 움직이는 머티리얼이 여기에 해당합니다.

해당 에셋: `SM_waterfall` 1개 (나머지 242개는 안정된 뒤 저장)

## 실행 명령
```
UnrealEditor.exe "C:/Unreal Projects/LyraRPG/LyraRPG.uproject" -nosplash -NoLiveCoding -log=SB_Thumb.log -ExecCmds="py C:/Claude/SemanticBrowser/tools/extract_thumbs.py --out-dir C:/Claude/SemanticBrowser/thumbs --manifest C:/Claude/SemanticBrowser/manifest.csv --failed C:/Claude/SemanticBrowser/failed.csv --log-file C:/Claude/SemanticBrowser/thumbs/_extract_log.txt --resolution 512 --pitch -11.25 --yaw -157.5 --fov 30 --margin 1.05 --bg-linear 0.22 --sun-intensity 2 --sun-pitch -45 --sun-yaw -120 --sky-intensity 1 --exposure-bias 0 --min-frames 10 --check-every 5 --stable-needed 3 --max-frames 300 --quit-when-done"
```

참고:
- `-ExecutePythonScript`를 쓰면 스크립트 실행 직후 에디터가 종료되어 틱 기반 처리가 진행되지 않습니다. 그래서 `-ExecCmds="py ..."`를 사용했습니다. `-ExecCmds`는 쉼표를 명령 구분자로 쓰므로, 인자 값에 쉼표를 넣으면 안 됩니다.
- 실행 일시: 2026-11-17 16:20 ~ 16:47 (UE 5.8, 별도 에디터 인스턴스)
