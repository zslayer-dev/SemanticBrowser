# -*- coding: utf-8 -*-
"""
에디터에서 문장으로 에셋 찾기 (UE 에디터 Python용, 표준 라이브러리만 사용)

준비: 먼저 검색 서버(serve.py)를 켜 둡니다.
  .venv/Scripts/python serve.py --index index.npz --manifest manifest.csv --port 8765 --default-k 10

사용: 에디터 Output Log 아래 입력칸을 "Python"이 아니라 "Cmd"로 두고:
  py "C:/Claude/SemanticBrowser/ue/sb.py" 길가에 세운 가로등
  py "C:/Claude/SemanticBrowser/ue/sb.py" 등불 --k 5
  py "C:/Claude/SemanticBrowser/ue/sb.py" wooden fence --select name     (이름 검색 결과를 선택)

하는 일
  1. 질의를 로컬 서버(127.0.0.1)로 보냅니다. 이미지·에셋은 보내지 않습니다(문장만).
  2. 결과를 Output Log에 표로 찍습니다: SigLIP 2 순위 + 이름 검색 순위.
  3. 콘텐츠 브라우저에서 결과 에셋들을 선택해 보여 줍니다(기본: SigLIP 결과).

참고: PoC 채점에서 한국어 임계값 구간이 매우 좁았으므로, 기본은 임계값 없이 상위 k개를 보여 줍니다.
"""

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

import unreal


def parse_args():
    p = argparse.ArgumentParser(prog="sb.py", description="SemanticBrowser 에디터 검색")
    p.add_argument("query", nargs="+", help="찾을 문장 (띄어쓰기 그대로 써도 됨)")
    p.add_argument("--k", type=int, default=0, help="결과 개수 (0 = 서버 기본값)")
    p.add_argument("--threshold", type=float, default=0.0, help="이 점수 미만 제외 (기본 0 = 사용 안 함)")
    p.add_argument("--server", default="http://127.0.0.1:8765", help="검색 서버 주소")
    p.add_argument("--timeout", type=float, default=30.0, help="서버 응답 대기(초)")
    p.add_argument("--select", choices=["siglip", "name", "none"], default="siglip",
                   help="콘텐츠 브라우저에서 선택할 결과")
    return p.parse_args(sys.argv[1:])


def main():
    a = parse_args()
    query = " ".join(a.query)   # 띄어쓰기로 나뉜 단어를 다시 한 문장으로

    # 내 컴퓨터 밖으로 질의를 보내지 않도록 서버 주소를 확인
    host = urllib.parse.urlparse(a.server).hostname
    if host not in ("127.0.0.1", "localhost", "::1"):
        unreal.log_error("[sb] 로컬 서버만 사용합니다: " + a.server)
        return

    params = {"q": query, "threshold": repr(a.threshold)}
    if a.k > 0:
        params["k"] = str(a.k)
    url = a.server.rstrip("/") + "/search?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=a.timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as e:
        unreal.log_error("[sb] 검색 서버에 연결하지 못했습니다 ({}). serve.py를 먼저 켜 주세요.".format(e))
        return

    unreal.log("[sb] 질의: {}  (k={}, threshold={})".format(data["query"], data["k"], data["threshold"]))
    unreal.log("[sb] --- SigLIP 2 (점수순) ---")
    if not data["results"]:
        unreal.log("[sb] (결과 없음)")
    for r in data["results"]:
        unreal.log("[sb] {:>2}  {:<40}  score={!r}".format(r["rank"], r["name"], r["score"]))
    unreal.log("[sb] --- 이름 검색 (겹치는 단어 수순) ---")
    if not data["name_matches"]:
        unreal.log("[sb] (결과 없음)")
    for r in data["name_matches"]:
        unreal.log("[sb] {:>2}  {:<40}  overlap={}".format(r["rank"], r["name"], r["overlap"]))

    # 콘텐츠 브라우저에서 선택 (여러 개를 한 번에 선택해 보여 줌)
    chosen = {"siglip": data["results"], "name": data["name_matches"], "none": []}[a.select]
    objs = [r["object_path"] for r in chosen if r.get("object_path")]
    if objs:
        unreal.EditorAssetLibrary.sync_browser_to_objects(objs)
        unreal.log("[sb] 콘텐츠 브라우저에서 {}개 선택 ({})".format(len(objs), a.select))


main()
