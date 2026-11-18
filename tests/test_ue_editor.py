# -*- coding: utf-8 -*-
"""
언리얼 에디터 통합 테스트 (실제 UE 5.8 에디터를 띄워서 확인)

에디터를 한 번 띄워서 아래 두 가지를 차례로 확인하고, 끝나면 에디터를 닫습니다(1~3분).
  1. 에디터 검색 (ue/sb.py → serve.py)
     - 테스트용 검색 서버를 빈 포트에 띄우고, 에디터 Output Log 명령으로 영어·한국어 질의를 보냄
     - 에디터 로그의 결과가 서버 결과(=search.py)와 같은지, 콘텐츠 브라우저 선택까지 실행됐는지
  2. 썸네일 추출 (tools/extract_thumbs.py)
     - thumbs/CONDITIONS.md의 '실행 명령'과 같은 조건으로 몇 개만 다시 찍음 (thumbs/_uetest_*/ 에 저장)
     - 크기·알파·배경색·잘림을 숫자로 확인하고, 기존 thumbs/ 의 PNG와 픽셀 단위로 같은지(재현성) 비교

필요한 것 (없으면 건너뜀)
  - 환경 변수 SB_UE_EDITOR  : UnrealEditor.exe 경로
  - 환경 변수 SB_UPROJECT   : LyraRPG.uproject 경로
  - index.npz, manifest.csv, thumbs/*.png (1·2단계 산출물)
  - 실행 중인 같은 프로젝트 에디터가 있어도 되지만, 무거우므로 닫아 두기를 권장

실행 예 (Git Bash):
  SB_UE_EDITOR="C:/Program Files/Epic Games/UE_5.8/Engine/Binaries/Win64/UnrealEditor.exe" \
  SB_UPROJECT="C:/Unreal Projects/LyraRPG/LyraRPG.uproject" \
  .venv/Scripts/python -m unittest tests.test_ue_editor -v
"""

import datetime
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import unittest
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDITOR = os.environ.get("SB_UE_EDITOR", "")
UPROJECT = os.environ.get("SB_UPROJECT", "")
PY = sys.executable

# 에디터 검색에 쓸 질의 (ExecCmds는 쉼표를 명령 구분자로 쓰므로 질의에 쉼표 금지)
QUERIES = ["wooden fence", "길가에 세운 가로등"]
# 썸네일 재현성 확인용 에셋 (움직이는 머티리얼인 SM_waterfall은 픽셀이 매번 달라서 제외)
THUMB_NAMES = ["SM_lantern_01", "SM_bamboo_02", "SM_wooden_fence_01"]
K = 3
EDITOR_TIMEOUT = 1200   # 초. 에디터 시작 + 썸네일 3개

READY = (os.path.isfile(EDITOR) and os.path.isfile(UPROJECT)
         and os.path.isfile(os.path.join(ROOT, "index.npz"))
         and os.path.isfile(os.path.join(ROOT, "manifest.csv"))
         and all(os.path.isfile(os.path.join(ROOT, "thumbs", n + ".png")) for n in THUMB_NAMES))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def conditions_args():
    """thumbs/CONDITIONS.md의 실행 명령에서 촬영 조건 인자만 뽑음 (출력 경로·종료 인자는 제외)."""
    with open(os.path.join(ROOT, "thumbs", "CONDITIONS.md"), encoding="utf-8") as f:
        text = f.read()
    cmd = re.search(r"extract_thumbs\.py (.*?)\"", text).group(1).split()
    drop_with_value = {"--out-dir", "--manifest", "--failed", "--log-file"}
    out, i = [], 0
    while i < len(cmd):
        if cmd[i] in drop_with_value:
            i += 2
            continue
        if cmd[i] != "--quit-when-done":
            out.append(cmd[i])
        i += 1
    return out


@unittest.skipUnless(READY, "SB_UE_EDITOR / SB_UPROJECT 환경 변수 또는 산출물(index.npz, manifest.csv, thumbs) 없음 → 건너뜀")
class TestInUnrealEditor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cls.port = free_port()
        cls.server_url = "http://127.0.0.1:{}".format(cls.port)
        cls.out_dir = os.path.join(ROOT, "thumbs", "_uetest_" + stamp).replace("\\", "/")
        os.makedirs(cls.out_dir)
        cls.log_name = "SB_Test_{}.log".format(stamp)
        cls.ue_log = os.path.join(os.path.dirname(UPROJECT), "Saved", "Logs", cls.log_name)

        # 1) 테스트용 검색 서버
        env = dict(os.environ, HF_HUB_DISABLE_SYMLINKS_WARNING="1", PYTHONIOENCODING="utf-8")
        cls.server = subprocess.Popen(
            [PY, "serve.py", "--index", "index.npz", "--manifest", "manifest.csv",
             "--port", str(cls.port), "--default-k", str(K)],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(120):
            try:
                urllib.request.urlopen(cls.server_url + "/health", timeout=1)
                break
            except Exception:
                time.sleep(1)
        else:
            cls.server.kill()
            raise RuntimeError("검색 서버가 시작되지 않음")

        # 2) 에디터 한 번 실행: 검색 명령들 → 썸네일 추출(끝나면 에디터 종료)
        sb = os.path.join(ROOT, "ue", "sb.py").replace("\\", "/")
        ex = os.path.join(ROOT, "tools", "extract_thumbs.py").replace("\\", "/")
        cmds = ["py {} {} --k {} --server {}".format(sb, q, K, cls.server_url) for q in QUERIES]
        cmds.append("py {} --names {} --out-dir {} --manifest {}/manifest.csv --failed {}/failed.csv "
                    "--log-file {}/log.txt {} --quit-when-done".format(
                        ex, "+".join(THUMB_NAMES), cls.out_dir, cls.out_dir, cls.out_dir, cls.out_dir,
                        " ".join(conditions_args())))
        try:
            subprocess.run([EDITOR, UPROJECT, "-nosplash", "-NoLiveCoding", "-log=" + cls.log_name,
                            "-ExecCmds=" + ", ".join(cmds)],
                           timeout=EDITOR_TIMEOUT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            cls.server.kill()   # 에디터가 끝나면 서버도 끔 (기대값은 아래에서 search.py로 직접 계산)
            cls.server.wait()
        with open(cls.ue_log, encoding="utf-8", errors="replace") as f:
            cls.log = f.read()

        # 서버를 다시 띄우지 않고도 비교할 수 있게, 기대값은 search.py로 직접 계산
        sys.path.insert(0, ROOT)
        from search import Searcher
        s = Searcher(os.path.join(ROOT, "index.npz"))
        cls.expected = {q: s.search(q, K) for q in QUERIES}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out_dir, ignore_errors=True)   # 테스트로 찍은 썸네일은 지움 (원본 thumbs/는 그대로)

    # ---- 1. 에디터 검색 -------------------------------------------------------
    def test_no_python_errors(self):
        errors = [l for l in self.log.splitlines() if "LogPython: Error" in l]
        self.assertEqual(errors, [])

    def test_search_results_match_search_py(self):
        lines = [l.split("LogPython: ", 1)[1] for l in self.log.splitlines() if "LogPython: [sb]" in l]
        for q in QUERIES:
            with self.subTest(query=q):
                # 이 질의의 블록: "[sb] 질의: q" 다음 줄부터 "--- 이름 검색" 전까지
                start = lines.index("[sb] 질의: {}  (k={}, threshold=0.0)".format(q, K))
                block = lines[start + 2: start + 2 + K]
                got = [(int(m.group(1)), m.group(2), float(m.group(3))) for m in
                       (re.match(r"\[sb\]\s+(\d+)\s+(\S+)\s+score=(\S+)", l) for l in block)]
                want = [(r, n, sc) for r, n, sc, _ in self.expected[q]]
                self.assertEqual([g[:2] for g in got], [w[:2] for w in want])
                for g, w in zip(got, want):
                    self.assertAlmostEqual(g[2] / w[2], 1.0, delta=1e-6)

    def test_content_browser_selection_ran(self):
        n = self.log.count("[sb] 콘텐츠 브라우저에서 {}개 선택 (siglip)".format(K))
        self.assertEqual(n, len(QUERIES))

    # ---- 2. 썸네일 추출 -------------------------------------------------------
    def test_thumbnails_written(self):
        with open(os.path.join(self.out_dir, "log.txt"), encoding="utf-8") as f:
            log = f.read()
        self.assertIn("완료: 성공 {} / 실패 0 / 대상 {}".format(len(THUMB_NAMES), len(THUMB_NAMES)), log)

    def test_thumbnail_numeric_conditions(self):
        for n in THUMB_NAMES:
            with self.subTest(name=n):
                a = np.asarray(Image.open(os.path.join(self.out_dir, n + ".png")).convert("RGBA")).astype(int)
                self.assertEqual(a.shape, (512, 512, 4))
                self.assertEqual(a[..., 3].min(), 255)                       # 투명 픽셀 없음
                # 네 모서리 = 배경. 기존 243장도 배경이 129~131로 ±1 흔들리므로 ±1 허용
                for y, x in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
                    self.assertLessEqual(int(np.abs(a[y, x, :3] - 130).max()), 1)
                obj = np.abs(a[..., :3] - 130).max(axis=2) > 12
                self.assertTrue(obj.any())                                   # 물체가 찍힘
                self.assertFalse(obj[0].any() or obj[-1].any() or obj[:, 0].any() or obj[:, -1].any())  # 안 잘림

    def test_thumbnails_reproduce_existing(self):
        for n in THUMB_NAMES:
            with self.subTest(name=n):
                new = np.asarray(Image.open(os.path.join(self.out_dir, n + ".png")))
                old = np.asarray(Image.open(os.path.join(ROOT, "thumbs", n + ".png")))
                diff = np.abs(new.astype(int) - old.astype(int))
                self.assertEqual(int(diff.max()), 0,
                                 "기존 썸네일과 다름: 최대 차이 {}, 다른 픽셀 비율 {}".format(
                                     int(diff.max()), float((diff.max(axis=2) > 0).mean())))


if __name__ == "__main__":
    unittest.main()
