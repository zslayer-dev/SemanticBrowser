# -*- coding: utf-8 -*-
"""
에디터 연동용 로컬 검색 서버 (serve.py)

왜 서버인가
  - UE에 들어 있는 Python에는 torch/transformers가 없습니다. 그래서 검색은 이 프로젝트의 .venv에서 돌리고,
    에디터는 HTTP로 질의만 보냅니다.
  - 모델을 불러오는 데 몇 초 걸리므로, 한 번 띄워 두고 계속 재사용합니다.

보안
  - 127.0.0.1(내 컴퓨터)에서만 접속을 받습니다. 다른 주소로 띄우려 하면 거부합니다.
  - 표준 라이브러리 http.server만 씁니다(추가 설치 없음).

요청
  GET /health                         → {"ok": true, "model": ..., "count": 243}
  GET /search?q=<문장>&k=10&threshold=0
      → {"query": ..., "results": [{"rank", "name", "score", "cosine", "asset_path", "object_path"}, ...],
         "name_matches": [{"rank", "name", "overlap", "asset_path", "object_path"}, ...]}
     - results: SigLIP 2 점수순 (search.py와 같은 계산, 반올림 없음)
     - name_matches: 이름 검색(4단계 기준선과 같은 방식: 질의 단어와 이름 '_' 토큰이 겹치는 수)

사용 예:
  .venv/Scripts/python serve.py --index index.npz --manifest manifest.csv --port 8765 --default-k 10
"""

import argparse
import csv
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from evaluate import baseline_rank   # 이름 검색은 채점 때와 똑같은 함수를 재사용
from search import Searcher

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def load_paths(manifest_path):
    """manifest.csv → {이름: (패키지 경로, 오브젝트 경로)}
    예: SM_lantern_01 → ("/Game/.../SM_lantern_01", "/Game/.../SM_lantern_01.SM_lantern_01")"""
    out = {}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pkg = r["asset_path"]
            out[r["asset_name"]] = (pkg, pkg + "." + r["asset_name"])
    return out


def make_handler(searcher, paths, default_k, max_k):
    lock = threading.Lock()   # 모델은 한 번에 한 질의만 처리 (동시 요청이 와도 안전하게)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            qs = parse_qs(u.query)
            if u.path == "/health":
                return self._send(200, {"ok": True, "model": searcher.model_name, "count": len(searcher.names)})
            if u.path != "/search":
                return self._send(404, {"error": "unknown path"})

            q = (qs.get("q") or [""])[0].strip()
            if not q:
                return self._send(400, {"error": "q(질의)가 비어 있습니다"})
            try:
                k = int((qs.get("k") or [default_k])[0])
                th = float((qs.get("threshold") or ["0"])[0])
            except ValueError:
                return self._send(400, {"error": "k 또는 threshold가 숫자가 아닙니다"})
            k = max(1, min(k, max_k))

            with lock:
                res = searcher.search(q, k, th)
            results = [{"rank": r, "name": n, "score": s, "cosine": c,
                        "asset_path": paths.get(n, ("", ""))[0], "object_path": paths.get(n, ("", ""))[1]}
                       for r, n, s, c in res]
            ranked, overlap = baseline_rank(q, searcher.names)
            name_matches = [{"rank": i + 1, "name": n, "overlap": overlap[n],
                             "asset_path": paths.get(n, ("", ""))[0], "object_path": paths.get(n, ("", ""))[1]}
                            for i, n in enumerate(ranked[:k])]
            self._send(200, {"query": q, "k": k, "threshold": th,
                             "results": results, "name_matches": name_matches})

        def log_message(self, fmt, *args):   # 접속 로그를 짧게
            sys.stderr.write("[serve] " + (fmt % args) + "\n")

    return Handler


def main():
    p = argparse.ArgumentParser(description="SemanticBrowser 로컬 검색 서버 (127.0.0.1 전용)")
    p.add_argument("--index", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--host", default="127.0.0.1", help="127.0.0.1 / localhost / ::1 만 허용")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--default-k", type=int, required=True, help="k를 안 주면 돌려줄 개수")
    p.add_argument("--max-k", type=int, default=50, help="한 번에 돌려줄 최대 개수")
    a = p.parse_args()

    if a.host not in LOOPBACK:
        sys.exit("외부에서 접속 가능한 주소로는 띄우지 않습니다: " + a.host)

    print("모델·색인 로드 중...")
    s = Searcher(a.index)
    paths = load_paths(a.manifest)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(s, paths, a.default_k, a.max_k))
    print("준비 완료: http://{}:{}/search?q=등불  (끄려면 Ctrl+C)".format(a.host, a.port))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
