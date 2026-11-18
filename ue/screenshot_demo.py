# -*- coding: utf-8 -*-
"""
에디터 검색 화면 스크린샷 찍기 (UE 에디터 Python, 표준 라이브러리만 사용)

하는 일 (틱 콜백으로 한 단계씩 진행)
  1. Output Log 탭과 콘텐츠 브라우저 탭을 엽니다.
  2. 질의마다: ue/sb.py로 검색 → 콘텐츠 브라우저에서 결과 선택 → 썸네일이 그려질 때까지 대기
     → 'EditorShot All' 콘솔 명령으로 에디터 창을 BMP로 저장
     (EditorShot은 에디터 창만 찍습니다. 바탕화면이나 다른 프로그램은 찍히지 않습니다.)
  3. (--quit-when-done) 에디터 종료

BMP는 프로젝트 Saved/Screenshots/ 아래에 생깁니다. PNG 변환·복사는 바깥 Python에서 합니다.

사용 (검색 서버 serve.py를 먼저 켜 둔 상태에서):
  UnrealEditor.exe <uproject> -ExecCmds="py C:/Claude/SemanticBrowser/ue/screenshot_demo.py --queries 길가에 세운 가로등|나무 울타리 --k 5 --wait-frames 300 --quit-when-done"
  (-ExecCmds는 쉼표로 명령을 나누므로 질의 구분자는 '|')
"""

import argparse
import os
import sys

import unreal

SB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sb.py").replace("\\", "/")


def parse_args():
    p = argparse.ArgumentParser(prog="screenshot_demo.py")
    p.add_argument("--queries", required=True, nargs="+",
                   help="'|'로 구분한 질의들 (띄어쓰기 그대로 가능)")
    p.add_argument("--k", type=int, required=True, help="질의마다 결과 개수")
    p.add_argument("--server", default="http://127.0.0.1:8765", help="검색 서버 주소")
    p.add_argument("--wait-frames", type=int, required=True,
                   help="검색 후 찍기까지 기다릴 프레임 수 (콘텐츠 브라우저 썸네일이 그려질 시간)")
    p.add_argument("--quit-when-done", action="store_true")
    a = p.parse_args(sys.argv[1:])
    a.queries = [q.strip() for q in " ".join(a.queries).split("|") if q.strip()]
    return a


class Demo:
    def __init__(self, a):
        self.a = a
        self.step = 0          # 지금 몇 번째 질의인지
        self.frames = 0        # 현재 단계에서 지난 프레임
        self.phase = "open"    # open → search → wait → shot → ... → done
        self.handle = unreal.register_slate_post_tick_callback(self.tick)

    def cmd(self, text):
        unreal.SystemLibrary.execute_console_command(None, text)

    def tick(self, dt):
        try:
            self._tick()
        except Exception as e:
            unreal.log_error("[shot] 예외로 중단: {!r}".format(e))
            self.finish()

    def _tick(self):
        self.frames += 1
        if self.phase == "open":
            # 에디터가 다 뜬 뒤(120프레임 후) 탭을 엶.
            # 주의: spawn_registered_tab_by_id는 탭을 열어도 '에디터 유틸리티 위젯 탭'이 아니면 False를 돌려줍니다
            # (엔진 구현). 실제 실패는 로그에 "TryInvokeTab failed"로 남으므로 반환값은 보지 않습니다.
            if self.frames < 120:
                return
            sub = unreal.get_editor_subsystem(unreal.EditorUtilitySubsystem)
            for tab in ("ContentBrowserTab1", "OutputLog"):
                sub.spawn_registered_tab_by_id(tab)
                unreal.log("[shot] 탭 열기 호출: " + tab)
            self.phase, self.frames = "search", 0
        elif self.phase == "search" and self.frames > 30:
            q = self.a.queries[self.step]
            self.cmd('py "{}" {} --k {} --server {}'.format(SB, q, self.a.k, self.a.server))
            self.phase, self.frames = "wait", 0
        elif self.phase == "wait" and self.frames >= self.a.wait_frames:
            self.cmd("EditorShot All")
            unreal.log("[shot] 스크린샷 {}/{}: {}".format(self.step + 1, len(self.a.queries), self.a.queries[self.step]))
            self.step += 1
            self.phase, self.frames = ("search" if self.step < len(self.a.queries) else "done"), 0
        elif self.phase == "done" and self.frames > 30:
            self.finish()

    def finish(self):
        if self.handle is not None:
            unreal.unregister_slate_post_tick_callback(self.handle)
            self.handle = None
        unreal.log("[shot] 완료")
        if self.a.quit_when_done:
            unreal.SystemLibrary.quit_editor()


_demo = Demo(parse_args())
