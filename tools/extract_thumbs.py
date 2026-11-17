# -*- coding: utf-8 -*-
"""
1단계: UE5 에디터 안에서 StaticMesh 썸네일을 PNG로 뽑는 스크립트.

[이 파일은 일반 Python이 아니라 "언리얼 에디터 안의 Python"에서 실행합니다]
  - `import unreal` 은 에디터 안에서만 됩니다. 명령 프롬프트에서 python으로 돌리면 실패합니다.
  - 실행 방법 (둘 중 하나):
      (1) 열려 있는 에디터의 Output Log 창에서 Cmd 를 "Python"으로 두고:
          py "C:/Claude/SemanticBrowser/tools/extract_thumbs.py" --out-dir C:/Claude/SemanticBrowser/thumbs
      (2) 에디터를 하나 더 띄워서 자동 실행 (끝나면 스스로 종료):
          UnrealEditor.exe "<프로젝트>.uproject" -ExecutePythonScript="<이 파일> --out-dir ... --quit-when-done"

[어떻게 찍는가 — "직접 렌더링(B안)"]
  1. 아무것도 없는 임시 레벨(/Temp/...)을 새로 만듭니다. 저장하지 않으므로 프로젝트에 흔적이 남지 않습니다.
  2. 회색 배경용 큰 구(sphere)와 조명(디렉셔널 라이트 + 스카이 라이트)을 놓습니다.
  3. 메시 하나를 원점에 놓고, 메시 크기(바운딩 박스)에 맞춰 카메라 거리를 계산해
     항상 같은 각도·같은 화각(FOV)으로 SceneCapture2D(장면을 텍스처로 찍는 카메라)로 찍습니다.
  4. 셰이더/텍스처가 아직 로딩 중이면 그림이 바뀌므로, 같은 그림이 여러 번 연속으로 나올 때까지(=안정)
     기다렸다가 PNG로 저장합니다.
  5. 다음 메시로 바꿔서 반복합니다.

[왜 한 번에 for문으로 안 돌리고 "틱 콜백"을 쓰나]
  에디터는 한 프레임씩 그림을 그립니다. for문으로 243개를 한 번에 돌리면 그 사이에 프레임이 지나가지 않아서
  셰이더 컴파일·텍스처 로딩이 끝나지 않은 그림이 찍힐 수 있습니다.
  그래서 "에디터가 한 프레임 그릴 때마다 불리는 함수(틱 콜백)"를 등록해서 조금씩 진행합니다.

판정/조건에 쓰이는 숫자(해상도, 각도, 화각 등)는 전부 아래 argparse 인자로 받습니다. 코드에 박지 않습니다.
"""

import argparse   # 명령줄 인자를 읽는 표준 라이브러리
import csv        # manifest.csv / failed.csv 쓰기
import datetime   # 로그에 시각 찍기
import hashlib    # 그림이 바뀌었는지 비교하려고 픽셀 데이터를 해시(지문)로 만듦
import math       # sin, radians 등
import os         # 경로/파일 존재 확인
import sys        # sys.argv (UE가 py 명령 뒤의 인자를 여기에 넣어 줌)

import unreal     # 언리얼 에디터 Python API (에디터 안에서만 import 가능)


# ---------------------------------------------------------------------------
# 1) 인자 정의
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="StaticMesh 썸네일 PNG 추출 (UE 에디터 Python)")

    # --- 대상 ---
    p.add_argument("--content-path", default="/Game/Asian_Village",
                   help="찾을 콘텐츠 폴더 (하위 폴더 포함)")
    p.add_argument("--name-prefix", default="SM_",
                   help="이 접두사로 시작하는 이름만 대상")
    p.add_argument("--names", default="",
                   help="시험용: + 로 구분한 에셋 이름만 처리 (비우면 전부). -ExecCmds가 쉼표를 명령 구분자로 써서 쉼표는 못 씀")
    p.add_argument("--limit", type=int, default=0,
                   help="시험용: 앞에서부터 N개만 처리 (0 = 전부)")

    # --- 출력 ---
    p.add_argument("--out-dir", required=True, help="PNG를 저장할 폴더 (thumbs)")
    p.add_argument("--manifest", required=True, help="manifest.csv 경로")
    p.add_argument("--failed", required=True, help="failed.csv 경로")
    p.add_argument("--log-file", default="", help="진행 로그를 남길 파일 (비우면 UE 로그에만)")

    # --- 촬영 조건 (전 에셋 동일) ---
    p.add_argument("--resolution", type=int, required=True, help="정사각형 한 변 픽셀 수")
    p.add_argument("--pitch", type=float, required=True, help="카메라 피치(도). 음수 = 내려다봄")
    p.add_argument("--yaw", type=float, required=True, help="카메라 요(도)")
    p.add_argument("--fov", type=float, required=True, help="카메라 수평 화각(도)")
    p.add_argument("--margin", type=float, required=True,
                   help="여백 배율. 1.0 = 바운딩 박스 꼭짓점이 화면 끝에 닿는 거리, 클수록 작게 보임")
    p.add_argument("--bg-linear", type=float, required=True,
                   help="배경 구의 발광 밝기 (선형값, R=G=B). 톤매핑 후 실제 PNG 값은 CONDITIONS.md에 측정해서 적음")
    p.add_argument("--sun-intensity", type=float, required=True, help="디렉셔널 라이트 세기(lux)")
    p.add_argument("--sun-pitch", type=float, required=True, help="디렉셔널 라이트 피치(도)")
    p.add_argument("--sun-yaw", type=float, required=True, help="디렉셔널 라이트 요(도)")
    p.add_argument("--sky-intensity", type=float, required=True, help="스카이 라이트 세기 배율")
    p.add_argument("--exposure-bias", type=float, required=True,
                   help="수동 노출 보정값(EV). 자동 노출은 끔")

    # --- 안정화 대기 ---
    p.add_argument("--min-frames", type=int, required=True,
                   help="메시를 바꾼 뒤 최소 이만큼 프레임을 기다린 후 비교 시작")
    p.add_argument("--check-every", type=int, required=True, help="몇 프레임마다 찍어서 비교할지")
    p.add_argument("--stable-needed", type=int, required=True,
                   help="연속으로 같은 그림이 이만큼 나오면 '안정'으로 보고 저장")
    p.add_argument("--max-frames", type=int, required=True,
                   help="이 프레임까지 안정되지 않으면 그 시점 그림을 저장하고 note에 표시")

    p.add_argument("--quit-when-done", action="store_true", help="끝나면 에디터 종료 (자동 실행용)")

    # UE의 py 명령은 sys.argv[0]에 스크립트 경로, 그 뒤에 인자를 넣어 줍니다.
    return p.parse_args(sys.argv[1:])


# ---------------------------------------------------------------------------
# 2) 작은 도우미 함수들
# ---------------------------------------------------------------------------
class Logger:
    """UE 출력 로그와 (지정했다면) 파일에 동시에 쓰는 간단한 로거."""

    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8") if path else None

    def __call__(self, msg):
        line = "[{}] {}".format(datetime.datetime.now().strftime("%H:%M:%S"), msg)
        unreal.log("[extract_thumbs] " + line)
        if self.f:
            self.f.write(line + "\n")
            self.f.flush()   # 중간에 에디터가 죽어도 로그가 남도록 바로 기록

    def close(self):
        if self.f:
            self.f.close()


def find_targets(content_path, prefix, log):
    """content_path 아래에서 이름이 prefix로 시작하는 StaticMesh 에셋 경로 목록을 이름순으로 돌려줌."""
    reg = unreal.AssetRegistryHelpers.get_asset_registry()
    # recursive=True: 하위 폴더까지. 에셋 레지스트리는 에셋을 실제로 로드하지 않고 목록만 줍니다.
    datas = reg.get_assets_by_path(content_path, recursive=True)
    out = []
    for d in datas:
        cls = str(d.asset_class_path.asset_name)        # 예: "StaticMesh"
        name = str(d.asset_name)                        # 예: "SM_lantern_01"
        if cls == "StaticMesh" and name.startswith(prefix):
            # package_name 예: /Game/Asian_Village/meshes/props/SM_lantern_01
            out.append((name, str(d.package_name)))
    out.sort(key=lambda t: t[0])  # 이름순 → 실행할 때마다 순서가 같음
    log("대상 StaticMesh {}개 발견 ({} 아래, 접두사 {})".format(len(out), content_path, prefix))
    return out


def make_backdrop_material(bg_linear):
    """
    배경 구에 쓸 '조명 영향을 받지 않는(Unlit) 단색' 머티리얼을 메모리에만 만듭니다.
    transient package(임시 꾸러미)에 만들기 때문에 프로젝트 콘텐츠 폴더에 파일이 생기지 않습니다.
    """
    mel = unreal.MaterialEditingLibrary
    # outer(소속)를 지정하지 않으면 UE가 임시(transient) 꾸러미에 만들어 줍니다.
    mat = unreal.new_object(unreal.Material, name="SB_BackdropMat")
    mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)  # 조명 무시
    mat.set_editor_property("two_sided", True)  # 구 '안쪽'에서 보므로 뒷면도 그려야 함
    # 상수 색 노드 하나 만들어서 Emissive(발광) 입력에 연결
    c = mel.create_material_expression(mat, unreal.MaterialExpressionConstant3Vector, -300, 0)
    c.set_editor_property("constant", unreal.LinearColor(bg_linear, bg_linear, bg_linear, 1.0))
    mel.connect_material_property(c, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    mel.recompile_material(mat)  # 셰이더 컴파일 요청 (비동기라 안정화 대기에서 기다림)
    return mat


def pixels_digest(world, rt):
    """렌더 타깃의 픽셀을 읽어 해시(지문) 문자열로. 그림이 조금이라도 바뀌면 값이 달라짐."""
    colors = unreal.RenderingLibrary.read_render_target(world, rt, False)  # Color 목록 (0~255)
    h = hashlib.md5()
    h.update(bytes(v for c in colors for v in (c.r, c.g, c.b, c.a)))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 3) 본체: 틱마다 조금씩 진행하는 상태 기계
# ---------------------------------------------------------------------------
class ThumbJob:
    def __init__(self, args):
        self.a = args
        self.log = Logger(args.log_file)
        os.makedirs(args.out_dir, exist_ok=True)

        self.targets = find_targets(args.content_path, args.name_prefix, self.log)
        if args.names:
            wanted = set(n.strip() for n in args.names.split("+") if n.strip())
            self.targets = [t for t in self.targets if t[0] in wanted]
            self.log("--names → {}개만 처리".format(len(self.targets)))
        if args.limit > 0:
            self.targets = self.targets[: args.limit]
            self.log("--limit {} → {}개만 처리".format(args.limit, len(self.targets)))

        self.manifest_rows = []   # 성공: (이름, 에셋 경로, png 경로)
        self.failed_rows = []     # 실패: (이름, 에셋 경로, 이유)
        self.idx = -1             # 지금 처리 중인 대상 번호
        self.frames = 0           # 현재 메시로 바꾼 뒤 지난 프레임 수
        self.prev_digest = None   # 직전 비교 때의 픽셀 지문
        self.stable = 0           # 연속으로 같았던 횟수
        self.phase = "warmup"     # warmup(배경·조명 준비) → mesh(메시 촬영) → done
        self.handle = None

        self._setup_scene()

    # --- 장면 만들기 -------------------------------------------------------
    def _setup_scene(self):
        a = self.a
        les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
        # 빈 임시 레벨. /Temp/ 경로는 디스크에 저장되지 않는 임시 영역입니다.
        # 이름이 겹치면 거부되므로 실행 시각을 붙여 매번 다른 이름으로 만듭니다.
        level_path = "/Temp/SB_ThumbLevel_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not les.new_level(level_path):
            raise RuntimeError("임시 레벨을 만들지 못했습니다: " + level_path)
        self.log("임시 레벨 생성: " + level_path)
        self.world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
        eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

        # 텍스처 스트리밍 끄기: 켜져 있으면 처음엔 저해상도 텍스처가 보이다가 나중에 선명해짐.
        # 끄면 모든 텍스처를 최고 해상도로 올립니다. (끝나면 원래 값으로 되돌림)
        self.prev_streaming = unreal.SystemLibrary.get_console_variable_int_value("r.TextureStreaming")
        unreal.SystemLibrary.execute_console_command(self.world, "r.TextureStreaming 0")

        # (1) 배경 구: 엔진 기본 구(반지름 50cm)를 메시마다 크기를 맞춰 키움
        self.backdrop = eas.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(0, 0, 0))
        bcomp = self.backdrop.static_mesh_component
        bcomp.set_static_mesh(unreal.load_asset("/Engine/BasicShapes/Sphere"))
        bcomp.set_material(0, make_backdrop_material(a.bg_linear))
        bcomp.set_editor_property("cast_shadow", False)

        # (2) 해 역할 디렉셔널 라이트 (방향만 의미 있음, 위치 무관)
        sun = eas.spawn_actor_from_class(
            unreal.DirectionalLight, unreal.Vector(0, 0, 0),
            unreal.Rotator(roll=0.0, pitch=a.sun_pitch, yaw=a.sun_yaw))
        sun.light_component.set_intensity(a.sun_intensity)

        # (3) 스카이 라이트: 주변(=회색 배경 구)을 찍어서 은은한 전체 조명으로 사용
        self.sky = eas.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(0, 0, 0))
        skc = self.sky.light_component
        skc.set_editor_property("source_type", unreal.SkyLightSourceType.SLS_CAPTURED_SCENE)
        skc.set_intensity(a.sky_intensity)

        # (4) 촬영 대상 메시를 올릴 액터 (메시만 바꿔 끼우며 재사용)
        self.subject = eas.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(0, 0, 0))
        scomp = self.subject.static_mesh_component
        scomp.set_editor_property("forced_lod_model", 1)  # 1 = LOD0(가장 자세한 단계) 고정
        # WPO(World Position Offset) 끄기: 나무·대나무 머티리얼의 '바람 흔들림'은 시간에 따라
        # 정점을 움직여서 매 프레임 그림이 달라짐 → 같은 조건으로 찍을 수 없음. 꺼서 멈춘 모습으로 찍음.
        scomp.set_editor_property("evaluate_world_position_offset", False)

        # (5) 카메라 역할 SceneCapture2D + 결과를 받을 렌더 타깃(텍스처)
        self.rt = unreal.RenderingLibrary.create_render_target2d(
            self.world, a.resolution, a.resolution,
            unreal.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
        self.cap = eas.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(0, 0, 0))
        cc = self.cap.capture_component2d
        cc.set_editor_property("texture_target", self.rt)
        # FinalColorLDR = 톤매핑까지 끝난, 눈에 보이는 그대로의 색
        cc.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
        cc.set_editor_property("fov_angle", a.fov)
        cc.set_editor_property("capture_every_frame", False)   # 우리가 원할 때만 찍음
        cc.set_editor_property("capture_on_movement", False)
        # 배경 구와 대상 메시만 찍기 (다른 건 무시)
        cc.set_editor_property("primitive_render_mode",
                               unreal.SceneCapturePrimitiveRenderMode.PRM_USE_SHOW_ONLY_LIST)
        # (속성 직접 대입은 막혀 있어서, 같은 일을 하는 함수로 등록)
        cc.show_only_actor_components(self.backdrop)
        cc.show_only_actor_components(self.subject)
        # 프레임마다 결과가 달라지는 효과는 끔 → 같은 입력이면 같은 그림
        flags = []
        for name in ["TemporalAA", "MotionBlur", "Fog", "VolumetricFog", "AtmosphericFog",
                     "Atmosphere", "Bloom", "LensFlares", "EyeAdaptation", "Grain"]:
            s = unreal.EngineShowFlagsSetting()
            s.set_editor_property("show_flag_name", name)
            s.set_editor_property("enabled", False)
            flags.append(s)
        cc.set_editor_property("show_flag_settings", flags)

        # 후처리: 자동 노출 끄고 고정 노출, Lumen(GI/반사) 끔 → 매번 같은 밝기
        pp = cc.get_editor_property("post_process_settings")
        pp.set_editor_property("override_auto_exposure_method", True)
        pp.set_editor_property("auto_exposure_method", unreal.AutoExposureMethod.AEM_MANUAL)
        pp.set_editor_property("override_auto_exposure_apply_physical_camera_exposure", True)
        pp.set_editor_property("auto_exposure_apply_physical_camera_exposure", False)
        pp.set_editor_property("override_auto_exposure_bias", True)
        pp.set_editor_property("auto_exposure_bias", a.exposure_bias)
        pp.set_editor_property("override_dynamic_global_illumination_method", True)
        pp.set_editor_property("dynamic_global_illumination_method",
                               unreal.DynamicGlobalIlluminationMethod.NONE)
        pp.set_editor_property("override_reflection_method", True)
        pp.set_editor_property("reflection_method", unreal.ReflectionMethod.NONE)
        pp.set_editor_property("override_vignette_intensity", True)
        pp.set_editor_property("vignette_intensity", 0.0)
        cc.set_editor_property("post_process_settings", pp)

        # 준비 단계용: 배경만 보이게 대상 메시를 비워 두고 카메라는 원점 근처에 둠
        self._frame_object(unreal.Vector(0, 0, 0), unreal.Vector(50, 50, 50))
        self.log("장면 준비 완료. 배경 셰이더·스카이 라이트 안정화 대기")

    def _frame_object(self, center, extent):
        """
        바운딩 박스(중심 center, 반쪽 크기 extent)의 꼭짓점 8개가 모두 화면 안에 들어오는
        '가장 가까운 거리'를 구하고, 거기에 margin을 곱해 카메라를 놓습니다.
        → 물체가 잘리지 않으면서 화면을 최대한 채움. 카메라는 항상 박스 중심을 바라봄.
        """
        a = self.a
        rot = unreal.Rotator(roll=0.0, pitch=a.pitch, yaw=a.yaw)   # 카메라가 바라보는 방향
        fwd = rot.get_forward_vector()    # 카메라 앞 방향
        right = rot.get_right_vector()    # 카메라 화면의 오른쪽
        up = rot.get_up_vector()          # 카메라 화면의 위쪽
        t = math.tan(math.radians(a.fov) / 2.0)   # 화면 끝까지의 기울기 (정사각형이라 가로=세로)

        need = 0.0
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    # 박스 중심에서 꼭짓점까지의 벡터
                    o = unreal.Vector(sx * extent.x, sy * extent.y, sz * extent.z)
                    z = o.dot(fwd)     # 카메라 앞쪽으로 얼마나 나와 있나
                    x = o.dot(right)   # 화면 좌우 위치
                    y = o.dot(up)      # 화면 상하 위치
                    # 꼭짓점이 화면 안에 있으려면 |x| <= t * (거리 + z) → 거리 >= |x|/t - z (y도 동일)
                    need = max(need, abs(x) / t - z, abs(y) / t - z)
        dist = need * a.margin
        radius = extent.length()
        cam_loc = center - fwd * dist   # 중심에서 '바라보는 방향의 반대'로 dist만큼 물러남
        self.cap.set_actor_location_and_rotation(cam_loc, rot, False, False)
        # 배경 구는 카메라와 메시를 넉넉히 감싸도록: (거리+반지름)의 3배, 구 원래 반지름이 50
        s = (dist + radius) * 3.0 / 50.0
        self.backdrop.set_actor_location(center, False, False)
        self.backdrop.set_actor_scale3d(unreal.Vector(s, s, s))

    # --- 매 프레임 불리는 함수 ---------------------------------------------
    def tick(self, delta_seconds):
        try:
            self._tick()
        except Exception as e:  # 예외가 나면 틱 콜백이 계속 에러를 뿜지 않도록 정리
            self.log("예외로 중단: {!r}".format(e))
            self._finish()

    def _tick(self):
        a = self.a
        self.frames += 1

        if self.phase == "warmup":
            # 처음 몇 프레임 뒤 스카이 라이트가 회색 배경을 한 번 찍도록 함
            if self.frames == a.min_frames:
                self.sky.light_component.recapture_sky()
            if self._stable_now():
                self.log("배경 안정화 완료 ({}프레임)".format(self.frames))
                self._next_target()
            elif self.frames >= a.max_frames:
                raise RuntimeError("배경이 {}프레임 안에 안정되지 않음".format(a.max_frames))
            return

        if self.phase != "mesh":
            return

        name, pkg = self.targets[self.idx]
        if self._stable_now():
            self._save(name, pkg, "stable")
        elif self.frames >= a.max_frames:
            # 물 흐름처럼 '시간에 따라 계속 움직이는 머티리얼'은 영원히 안정되지 않습니다.
            # 이 시점이면 셰이더·텍스처 로딩은 끝났으므로, 지금 그림을 찍어 저장하되 note로 표시합니다.
            self.cap.capture_component2d.capture_scene()
            self._save(name, pkg, "unstable_at_max_frames")

    def _save(self, name, pkg, note):
        """렌더 타깃을 PNG로 저장하고 결과를 manifest/failed 목록에 기록한 뒤 다음 대상으로."""
        png = os.path.join(self.a.out_dir, name + ".png").replace("\\", "/")
        unreal.RenderingLibrary.export_render_target(self.world, self.rt, self.a.out_dir, name + ".png")
        if os.path.isfile(png):
            self.manifest_rows.append((name, pkg, png, note))
            self.log("[{}/{}] 저장 {} ({}프레임, {})".format(
                self.idx + 1, len(self.targets), name, self.frames, note))
        else:
            self.failed_rows.append((name, pkg, "PNG 저장 실패(파일 없음)"))
            self.log("[{}/{}] 실패 {}: PNG 저장 실패".format(self.idx + 1, len(self.targets), name))
        self._next_target()

    def _stable_now(self):
        """check_every 프레임마다 찍어서 직전과 같으면 stable+1. stable_needed 도달 시 True."""
        a = self.a
        if self.frames < a.min_frames or (self.frames - a.min_frames) % a.check_every != 0:
            return False
        self.cap.capture_component2d.capture_scene()
        d = pixels_digest(self.world, self.rt)
        self.stable = self.stable + 1 if d == self.prev_digest else 0
        self.prev_digest = d
        return self.stable >= a.stable_needed

    def _next_target(self):
        """다음 대상으로 넘어감. 이미 PNG가 있으면 건너뜀(덮어쓰지 않음). 로드 실패는 failed에 기록."""
        while True:
            self.idx += 1
            self.frames, self.prev_digest, self.stable = 0, None, 0
            if self.idx >= len(self.targets):
                self._finish()
                return
            name, pkg = self.targets[self.idx]
            png = os.path.join(self.a.out_dir, name + ".png").replace("\\", "/")
            if os.path.isfile(png):
                self.manifest_rows.append((name, pkg, png, "existing_skipped"))
                self.log("[{}/{}] 이미 있음, 건너뜀 {}".format(self.idx + 1, len(self.targets), name))
                continue
            mesh = unreal.load_asset(pkg)
            if mesh is None or not isinstance(mesh, unreal.StaticMesh):
                self.failed_rows.append((name, pkg, "에셋 로드 실패"))
                self.log("[{}/{}] 실패 {}: 로드 실패".format(self.idx + 1, len(self.targets), name))
                continue
            b = mesh.get_bounds()            # 메시 자체 좌표 기준 경계 (원점에 놓으므로 그대로 사용)
            if b.box_extent.length() <= 0:
                self.failed_rows.append((name, pkg, "바운딩 크기 0"))
                self.log("[{}/{}] 실패 {}: 바운딩 0".format(self.idx + 1, len(self.targets), name))
                continue
            self.subject.static_mesh_component.set_static_mesh(mesh)
            self._frame_object(b.origin, b.box_extent)
            self.phase = "mesh"
            return

    def _finish(self):
        self.phase = "done"
        if self.handle is not None:
            unreal.unregister_slate_post_tick_callback(self.handle)
            self.handle = None
        unreal.SystemLibrary.execute_console_command(
            self.world, "r.TextureStreaming {}".format(self.prev_streaming))

        # manifest/failed는 "현재 thumbs 상태 목록"이라 실행마다 새로 씁니다(이름순).
        with open(self.a.manifest, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            # note: stable(안정 후 저장) / unstable_at_max_frames(움직이는 머티리얼) / existing_skipped(이미 있어 건너뜀)
            w.writerow(["asset_name", "asset_path", "png_path", "note"])
            w.writerows(sorted(self.manifest_rows))
        with open(self.a.failed, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["asset_name", "asset_path", "reason"])
            w.writerows(sorted(self.failed_rows))
        self.log("완료: 성공 {} / 실패 {} / 대상 {}".format(
            len(self.manifest_rows), len(self.failed_rows), len(self.targets)))
        self.log.close()
        if self.a.quit_when_done:
            unreal.SystemLibrary.quit_editor()

    def start(self):
        self.handle = unreal.register_slate_post_tick_callback(self.tick)


# 전역 변수에 붙잡아 둬야 함수가 끝나도 작업 객체가 사라지지 않음
_args = parse_args()
try:
    _job = ThumbJob(_args)
    _job.start()
except Exception:
    # 준비 단계에서 실패하면 오류를 로그에 남기고, 자동 실행이면 에디터를 닫아 멈춰 있지 않게 함
    import traceback
    unreal.log_error("[extract_thumbs] 준비 실패:\n" + traceback.format_exc())
    if _args.quit_when_done:
        unreal.SystemLibrary.quit_editor()
    raise
