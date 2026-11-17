# -*- coding: utf-8 -*-
"""
4단계: 채점 (evaluate.py)

answer_key.csv의 질의 13건(정답 있음 10, 없음 3)을 한국어·영어 따로 돌려서 아래를 계산합니다.

(1) SigLIP 2 결과 (질의마다)
    - 통과: 정답 집합 중 하나라도 상위 top-k(기본 5) 안에 있는가
    - 첫 정답의 순위와 점수 (전체 243개 중 순위, top-k 밖이어도 기록)
    - 상위 top-k 안의 함정(traps) 개수
    - absent 질의는 1위 점수
(2) 기준선 — 이름 검색 (영어 질의만)
    - 질의를 소문자 단어로, 에셋 이름을 소문자 '_' 토큰으로 나눠 겹치는 단어 수로 정렬
      (동률은 이름순, 겹치는 단어 0개는 결과에서 제외)
    - SigLIP과 같은 기준(통과 여부, 함정 개수)으로 채점
(3) 임계값 구간 (한국어·영어 각각)
    - 임계값 t를 로그 간격으로 훑으며
        absent 질의 중 '결과가 빈' 개수 (1위 점수 < t)
        정답 질의 중 '상위 top-k 안의 정답 점수가 t 이상'인 개수
    - 구간 A: absent 전부 비고 정답 10건 모두 남는 t
      구간 B: absent 전부 비고 정답이 min-remaining(기본 8)건 이상 남는 t
    - 격자(훑은 점)에서의 범위와, 격자와 무관한 정확한 경계(점수에서 바로 계산)를 둘 다 적습니다.
    - 구간이 없으면 "없음"이라고 그대로 씁니다.

출력: results/<타임스탬프>/ 에 summary.md, per_query.csv, sweep.csv, report.html
점수는 반올림하지 않고 저장합니다. (summary.md 표도 전체 자릿수)

사용 예:
  .venv/Scripts/python evaluate.py --answer-key answer_key.csv --index index.npz --manifest manifest.csv \
      --results-dir results --top-k 5 --sweep-min 1e-6 --sweep-max 1 --sweep-n 61 --min-remaining 8
"""

import argparse
import csv
import datetime
import html
import math
import os
import re
import sys

import numpy as np

from search import Searcher


# ---------------------------------------------------------------------------
# 입력 읽기
# ---------------------------------------------------------------------------
def load_answer_key(path):
    """answer_key.csv → 질의 목록. answers/traps는 ';'로 나눈 이름 목록으로 바꿈."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "id": r["id"],
                "query_ko": r["query_ko"],
                "query_en": r["query_en"],
                "answers": [x for x in r["answers"].split(";") if x],
                "traps": [x for x in r["traps"].split(";") if x],
                "hard": r["hard"] == "1",
                "absent": r["absent"] == "1",
                "note": r["note"],
            })
    return rows


def load_manifest(path):
    """manifest.csv → {에셋 이름: PNG 경로}"""
    with open(path, newline="", encoding="utf-8") as f:
        return {r["asset_name"]: r["png_path"] for r in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# 순위 목록 하나를 채점하는 공통 함수 (SigLIP과 기준선이 똑같이 씀)
# ---------------------------------------------------------------------------
def grade(ranked, scores, q, top_k):
    """
    ranked: 순위대로 정렬된 에셋 이름 목록 (기준선은 '겹치는 단어 있는 것'만 들어 있음)
    scores: {이름: 점수} (SigLIP 점수 또는 기준선의 겹친 단어 수)
    q: 정답표 한 행
    """
    top = ranked[:top_k]
    answers, traps = set(q["answers"]), set(q["traps"])
    first_rank, first_name = None, ""
    for i, n in enumerate(ranked):
        if n in answers:
            first_rank, first_name = i + 1, n
            break
    traps_in = [n for n in top if n in traps]
    return {
        "pass": (any(n in answers for n in top)) if not q["absent"] else None,
        "first_answer_rank": first_rank,
        "first_answer_name": first_name,
        "first_answer_score": scores[first_name] if first_name else None,
        "traps_in_topk": len(traps_in),
        "trap_names_in_topk": ";".join(traps_in),
        "top1_name": ranked[0] if ranked else "",
        "top1_score": scores[ranked[0]] if ranked else None,
        "result_count": len(ranked),
        "topk_names": ";".join(top),
        "topk_scores": ";".join(repr(scores[n]) for n in top),
        # 임계값 계산용: 상위 top-k 안 정답들 중 가장 높은 점수 (없으면 None)
        "best_answer_score_in_topk": max((scores[n] for n in top if n in answers), default=None),
    }


# ---------------------------------------------------------------------------
# 기준선: 이름 검색 흉내
# ---------------------------------------------------------------------------
def baseline_rank(query_en, names):
    """
    영어 질의를 소문자 단어로 나누고(영문자·숫자 덩어리), 에셋 이름을 소문자 '_' 토큰으로 나눠
    '질의 단어 중 이름 토큰에 들어 있는 단어 수'로 정렬합니다. 동률은 이름순, 0개는 제외.
    """
    words = set(re.findall(r"[a-z0-9]+", query_en.lower()))
    scored = {}
    for n in names:
        tokens = set(n.lower().split("_"))
        k = len(words & tokens)
        if k > 0:
            scored[n] = k
    ranked = sorted(scored, key=lambda n: (-scored[n], n))
    return ranked, scored


# ---------------------------------------------------------------------------
# 임계값 훑기
# ---------------------------------------------------------------------------
def sweep(per_query, thresholds, n_answer_q, n_absent_q, min_remaining):
    """
    per_query: 한 언어의 SigLIP 채점 결과 목록
    반환: (격자 행 목록, 정확한 경계 dict)
    """
    absent_top1 = [r["top1_score"] for r in per_query if r["absent"]]
    ans_best = [r["best_answer_score_in_topk"] for r in per_query if not r["absent"]]

    rows = []
    for t in thresholds:
        absent_empty = sum(1 for s in absent_top1 if s < t)
        remain = sum(1 for s in ans_best if s is not None and s >= t)
        rows.append({
            "threshold": t,
            "absent_empty": absent_empty,
            "answer_remaining": remain,
            "A": absent_empty == n_absent_q and remain == n_answer_q,
            "B": absent_empty == n_absent_q and remain >= min_remaining,
        })

    # 정확한 경계 (격자와 무관):
    #   absent 전부 빔  ⇔ t > max(absent 1위 점수)
    #   정답 q가 남음  ⇔ t <= (q의 top-k 안 정답 최고점)
    #   → 정답이 m건 이상 남음 ⇔ t <= (정답 최고점들 중 m번째로 큰 값)
    lo = max(absent_top1) if absent_top1 else 0.0
    valid = sorted([s for s in ans_best if s is not None], reverse=True)

    def upper(m):
        return valid[m - 1] if len(valid) >= m else None

    exact = {"absent_max_top1": lo}
    for key, m in (("A", n_answer_q), ("B", min_remaining)):
        hi = upper(m)
        exact[key] = (lo, hi) if (hi is not None and hi > lo) else None
        exact[key + "_upper"] = hi
    return rows, exact


def grid_range(rows, key):
    """격자에서 조건을 만족한 t들의 (최소, 최대, 개수). 없으면 None."""
    ts = [r["threshold"] for r in rows if r[key]]
    return (min(ts), max(ts), len(ts)) if ts else None


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------
def fmt(v):
    """반올림 없이 그대로 (None은 빈칸)."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "O" if v else "X"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def write_report_html(path, results, manifest, out_dir, top_k):
    """질의마다 상위 top-k 썸네일과 점수. 정답=초록 테두리, 함정=빨간 테두리. 눈 검산용."""
    def img_cell(name, score, q):
        cls = "ans" if name in q["answers"] else ("trap" if name in q["traps"] else "")
        src = os.path.relpath(manifest[name], out_dir).replace("\\", "/") if name in manifest else ""
        return ('<figure class="{}"><img src="{}" alt="{}" loading="lazy">'
                '<figcaption>{}<br><code>{}</code></figcaption></figure>').format(
            cls, html.escape(src), html.escape(name), html.escape(name), html.escape(repr(score)))

    parts = ["""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>검색 결과 검산</title>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#ddd;--ans:#1a9e3a;--trap:#d22}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#eee;--muted:#aaa;--line:#333}}
body{background:var(--bg);color:var(--fg);font-family:system-ui,sans-serif;margin:16px}
section{border-top:1px solid var(--line);padding:12px 0}
h2{font-size:1rem;margin:0 0 6px}.meta{color:var(--muted);font-size:.85rem;margin-bottom:8px}
.row{display:flex;flex-wrap:wrap;gap:8px}
figure{margin:0;width:150px;border:4px solid transparent;font-size:.72rem;word-break:break-all}
figure img{width:100%;display:block}figure.ans{border-color:var(--ans)}figure.trap{border-color:var(--trap)}
.legend span{display:inline-block;padding:0 6px;margin-right:8px;border:3px solid}
</style></head><body>
<h1>검색 결과 검산 (상위 {k}개)</h1>
<p class="legend"><span style="border-color:var(--ans)">정답</span><span style="border-color:var(--trap)">함정</span>
점수는 반올림 없음. 썸네일은 로컬 thumbs/ 를 상대 경로로 참조합니다.</p>
""".replace("{k}", str(top_k))]
    for r in results:
        q = r["q"]
        for lang in ("ko", "en"):
            g = r[lang]
            label = "absent" if q["absent"] else ("통과" if g["pass"] else "실패")
            parts.append('<section><h2>{} [{}] {}</h2><div class="meta">{} · 첫 정답 순위 {} · 함정 {}개{}</div><div class="row">'.format(
                q["id"], lang, html.escape(q["query_" + lang]), label,
                fmt(g["first_answer_rank"]) or "-", g["traps_in_topk"],
                " · 정답: " + html.escape(";".join(q["answers"])) if q["answers"] else ""))
            names = g["topk_names"].split(";") if g["topk_names"] else []
            scores = g["topk_scores"].split(";") if g["topk_scores"] else []
            for n, s in zip(names, scores):
                parts.append(img_cell(n, float(s), q))
            parts.append("</div></section>")
    parts.append("</body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main():
    p = argparse.ArgumentParser(description="SigLIP 2 검색 채점")
    p.add_argument("--answer-key", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--manifest", required=True, help="report.html 썸네일 경로용")
    p.add_argument("--results-dir", required=True, help="이 아래에 <타임스탬프>/ 폴더를 새로 만듦")
    p.add_argument("--top-k", type=int, required=True, help="통과 판정·함정 집계 범위 (성공 기준: 5)")
    p.add_argument("--sweep-min", type=float, required=True, help="임계값 훑기 최소 (예: 1e-6)")
    p.add_argument("--sweep-max", type=float, required=True, help="임계값 훑기 최대 (예: 1)")
    p.add_argument("--sweep-n", type=int, required=True, help="훑을 점 개수 (로그 간격)")
    p.add_argument("--min-pass", type=int, required=True, help="성공 기준 1: 정답 질의 중 통과해야 할 수 (8)")
    p.add_argument("--min-remaining", type=int, required=True, help="구간 B: 남아야 할 정답 질의 수 (8)")
    p.add_argument("--max-length", type=int, default=64, help="텍스트 토큰 길이 (SigLIP 2 학습 조건 64)")
    p.add_argument("--no-lowercase", action="store_true")
    a = p.parse_args()

    # 실행마다 새 폴더 (덮어쓰기 금지)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(a.results_dir, stamp)
    if os.path.exists(out_dir):
        sys.exit("이미 있습니다: " + out_dir)
    os.makedirs(out_dir)

    key = load_answer_key(a.answer_key)
    manifest = load_manifest(a.manifest)
    s = Searcher(a.index, None, a.max_length, not a.no_lowercase)

    # 정답표의 이름이 색인에 다 있는지 먼저 확인 (오타가 있으면 채점이 조용히 틀어짐)
    missing = sorted({n for q in key for n in q["answers"] + q["traps"]} - set(s.names))
    if missing:
        sys.exit("정답표 이름 중 색인에 없는 것: {}".format(missing))

    results = []
    for q in key:
        r = {"q": q}
        for lang in ("ko", "en"):
            _, sc = s.scores(q["query_" + lang])
            ranked = sorted(s.names, key=lambda n, sc=sc, idx={n: i for i, n in enumerate(s.names)}: (-sc[idx[n]], n))
            scores = {n: float(sc[i]) for i, n in enumerate(s.names)}
            g = grade(ranked, scores, q, a.top_k)
            g["absent"] = q["absent"]
            r[lang] = g
        b_ranked, b_scores = baseline_rank(q["query_en"], s.names)
        r["base"] = grade(b_ranked, b_scores, q, a.top_k)
        results.append(r)
        print("{} 완료".format(q["id"]))

    answer_qs = [r for r in results if not r["q"]["absent"]]
    absent_qs = [r for r in results if r["q"]["absent"]]
    n_ans, n_abs = len(answer_qs), len(absent_qs)

    # 임계값 격자: 10^min ~ 10^max 를 로그 간격으로 n개
    thresholds = list(np.logspace(math.log10(a.sweep_min), math.log10(a.sweep_max), a.sweep_n))
    sweeps = {}
    for lang in ("ko", "en"):
        rows, exact = sweep([r[lang] for r in results], [float(t) for t in thresholds], n_ans, n_abs, a.min_remaining)
        sweeps[lang] = (rows, exact)

    # ---- per_query.csv ----
    fields = ["id", "lang", "method", "query", "hard", "absent", "pass", "first_answer_rank",
              "first_answer_name", "first_answer_score", "traps_in_topk", "trap_names_in_topk",
              "top1_name", "top1_score", "result_count", "topk_names", "topk_scores", "answers", "traps"]
    with open(os.path.join(out_dir, "per_query.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            q = r["q"]
            for lang, method, g in (("ko", "siglip2", r["ko"]), ("en", "siglip2", r["en"]), ("en", "name_baseline", r["base"])):
                row = {k: g.get(k) for k in fields if k in g}
                row.update({"id": q["id"], "lang": lang, "method": method, "query": q["query_" + lang],
                            "hard": int(q["hard"]), "absent": int(q["absent"]),
                            "answers": ";".join(q["answers"]), "traps": ";".join(q["traps"])})
                if row.get("pass") is not None:
                    row["pass"] = int(row["pass"])
                w.writerow({k: ("" if row.get(k) is None else (repr(row[k]) if isinstance(row[k], float) else row[k])) for k in fields})

    # ---- sweep.csv ----
    with open(os.path.join(out_dir, "sweep.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "ko_absent_empty", "ko_answer_remaining", "ko_A", "ko_B",
                    "en_absent_empty", "en_answer_remaining", "en_A", "en_B"])
        for rk, re_ in zip(sweeps["ko"][0], sweeps["en"][0]):
            w.writerow([repr(rk["threshold"]), rk["absent_empty"], rk["answer_remaining"], int(rk["A"]), int(rk["B"]),
                        re_["absent_empty"], re_["answer_remaining"], int(re_["A"]), int(re_["B"])])

    # ---- report.html ----
    write_report_html(os.path.join(out_dir, "report.html"), results, manifest, out_dir, a.top_k)

    # ---- summary.md ----
    def n_pass(lang):
        return sum(1 for r in answer_qs if r[lang]["pass"])

    def interval_text(lang, key_):
        rows, exact = sweeps[lang]
        g = grid_range(rows, key_)
        e = exact[key_]
        grid = "격자: **없음**" if g is None else "격자: {!r} ~ {!r} ({}개 점)".format(g[0], g[1], g[2])
        ex = "정확한 경계: **없음**" if e is None else "정확한 경계: {!r} < t ≤ {!r}".format(e[0], e[1])
        return "{} / {}".format(grid, ex)

    L = []
    L.append("# 채점 결과 ({})".format(stamp))
    L.append("")
    L.append("- 모델: `{}` (revision 은 index.npz 참고), logit_scale={!r}, logit_bias={!r}".format(s.model_name, s.logit_scale, s.logit_bias))
    L.append("- top-k={}, 임계값 격자 {!r} ~ {!r} 로그 간격 {}개, 질의: 정답 있음 {}건 / absent {}건".format(
        a.top_k, a.sweep_min, a.sweep_max, a.sweep_n, n_ans, n_abs))
    L.append("")
    L.append("## 성공 기준 (한국어 질의 기준, 영어는 비교용)")
    L.append("")
    L.append("| 기준 | 한국어 | 영어 |")
    L.append("|---|---|---|")
    L.append("| 1. 정답 질의 {}건 중 {}건 이상 top-{} 통과 (임계값 없음) | {} ({}/{}) | {} ({}/{}) |".format(
        n_ans, a.min_pass, a.top_k,
        "**통과**" if n_pass("ko") >= a.min_pass else "**실패**", n_pass("ko"), n_ans,
        "통과" if n_pass("en") >= a.min_pass else "실패", n_pass("en"), n_ans))
    b_ko, b_en = sweeps["ko"][1]["B"], sweeps["en"][1]["B"]
    L.append("| 2. 임계값 적용 시 absent {}건 결과 없음 (기준 1을 유지하는 임계값 = 구간 B 존재) | {} | {} |".format(
        n_abs, "**통과**" if b_ko else "**실패** (구간 B 없음)", "통과" if b_en else "실패 (구간 B 없음)"))
    L.append("")
    L.append("기준 2 해석: 임계값을 아주 높이면 absent는 항상 비지만 정답도 사라지므로, 기준 1(정답 {}건 이상 남음)을 함께 만족하는 임계값이 있는지(구간 B)로 판정했습니다.".format(a.min_remaining))
    L.append("")
    L.append("## SigLIP 2 vs 기준선(이름 검색)")
    L.append("")
    L.append("첫 정답 순위: 전체 목록에서 처음 나오는 정답의 순위 (기준선은 이름이 겹치는 결과 안에서만, 없으면 빈칸). 함정: top-{} 안의 함정 수.".format(a.top_k))
    L.append("")
    L.append("| id | 한국어 질의 | 영어 질의 | 어려움 | SigLIP 한국어 통과 | 첫 정답 순위 | 점수 | 함정 | SigLIP 영어 통과 | 첫 정답 순위 | 점수 | 함정 | 기준선 통과 | 첫 정답 순위 | 함정 | 결과 수 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in answer_qs:
        q, ko, en, b = r["q"], r["ko"], r["en"], r["base"]
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            q["id"], q["query_ko"], q["query_en"], "O" if q["hard"] else "",
            fmt(ko["pass"]), fmt(ko["first_answer_rank"]), fmt(ko["first_answer_score"]), ko["traps_in_topk"],
            fmt(en["pass"]), fmt(en["first_answer_rank"]), fmt(en["first_answer_score"]), en["traps_in_topk"],
            fmt(b["pass"]), fmt(b["first_answer_rank"]), b["traps_in_topk"], b["result_count"]))
    L.append("| 합계 | | | | {}/{} | | | {} | {}/{} | | | {} | {}/{} | | {} | |".format(
        n_pass("ko"), n_ans, sum(r["ko"]["traps_in_topk"] for r in answer_qs),
        n_pass("en"), n_ans, sum(r["en"]["traps_in_topk"] for r in answer_qs),
        sum(1 for r in answer_qs if r["base"]["pass"]), n_ans, sum(r["base"]["traps_in_topk"] for r in answer_qs)))
    L.append("")
    L.append("### absent 질의")
    L.append("")
    L.append("| id | 한국어 | 1위 (한국어) | 1위 점수 | 영어 | 1위 (영어) | 1위 점수 | 기준선 결과 수 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in absent_qs:
        q = r["q"]
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            q["id"], q["query_ko"], r["ko"]["top1_name"], fmt(r["ko"]["top1_score"]),
            q["query_en"], r["en"]["top1_name"], fmt(r["en"]["top1_score"]), r["base"]["result_count"]))
    L.append("")
    L.append("## 임계값 구간")
    L.append("")
    for lang, label in (("ko", "한국어"), ("en", "영어")):
        ex = sweeps[lang][1]
        L.append("### {}".format(label))
        L.append("")
        L.append("- absent 1위 점수 최대값: {!r} (t가 이보다 커야 absent {}건이 모두 빔)".format(ex["absent_max_top1"], n_abs))
        L.append("- 구간 A (absent 모두 빔, 정답 {}건 모두 남음): {}".format(n_ans, interval_text(lang, "A")))
        L.append("- 구간 B (absent 모두 빔, 정답 {}건 이상 남음): {}".format(a.min_remaining, interval_text(lang, "B")))
        L.append("")
    L.append("격자 결과는 sweep.csv, 질의별 상세는 per_query.csv, 썸네일 검산은 report.html (커밋 금지).")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    print("저장:", out_dir)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
