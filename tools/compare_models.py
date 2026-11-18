# -*- coding: utf-8 -*-
"""
모델 후보 비교 (tools/compare_models.py)

같은 썸네일 243장, 같은 정답표(answer_key.csv)로 여러 그림-문장 모델을 채점해 나란히 비교합니다.
4단계 evaluate.py의 채점 함수(grade, baseline_rank, sweep)를 그대로 재사용하므로 채점 기준이 같습니다.

후보마다 하는 일
  1. 썸네일 → 이미지 임베딩 (길이 1). 모델별 캐시 파일(--cache-dir/<모델>.npz)이 있으면 재사용, 없으면 새로 만듦
     (기존 캐시는 덮어쓰지 않음)
  2. 질의(한국어·영어) → 텍스트 임베딩 (길이 1)
     - SigLIP 계열: padding="max_length", max_length=64, 소문자 (search.py와 동일)
     - CLIP 계열: padding=True (CLIP 토크나이저가 소문자 처리를 스스로 함)
  3. 점수 = sigmoid(exp(logit_scale) × 코사인 + logit_bias)
     - CLIP에는 logit_bias가 없어 0으로 둡니다. CLIP은 '짝마다 맞다/아니다'가 아니라 '순위'를 학습한 모델이라
       이 점수는 1 쪽으로 몰리는 경향이 있습니다(그 자체가 비교 결과).
     - 임계값 구간은 이 점수로 한 번, 모델 점수의 형태와 무관하게 '코사인'으로도 한 번 계산합니다.
  4. 순위 지표(top-k 통과, 함정, 첫 정답 순위)와 임계값 구간을 계산

결과는 results/<타임스탬프>_models/ 새 폴더에 저장합니다 (summary.md, per_query.csv).

사용 예:
  .venv/Scripts/python tools/compare_models.py --manifest manifest.csv --answer-key answer_key.csv \
      --models google/siglip2-base-patch16-224 openai/clip-vit-base-patch32 Bingsu/clip-vit-base-patch32-ko \
      --cache-dir model_cache --results-dir results --top-k 5 --min-pass 8 --min-remaining 8 --batch-size 16
"""

import argparse
import csv
import datetime
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from evaluate import baseline_rank, grade, load_answer_key, sweep   # noqa: E402  채점 기준을 4단계와 똑같이


def to_tensor(out):
    """transformers 버전에 따라 텐서 또는 출력 객체 → 텐서."""
    return out if isinstance(out, torch.Tensor) else out.pooler_output


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class Candidate:
    """후보 모델 하나: 이미지·텍스트 임베딩과 점수 계산."""

    def __init__(self, name):
        self.name = name
        self.model = AutoModel.from_pretrained(name).eval()
        self.processor = AutoProcessor.from_pretrained(name)
        self.kind = self.model.config.model_type            # 'siglip' / 'clip' 등
        self.scale = float(self.model.logit_scale.detach())
        bias = getattr(self.model, "logit_bias", None)
        self.bias = float(bias.detach()) if bias is not None else 0.0   # CLIP은 bias 없음 → 0

    def image_embeddings(self, paths, batch_size):
        feats = []
        with torch.no_grad():
            for i in range(0, len(paths), batch_size):
                imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
                inp = self.processor(images=imgs, return_tensors="pt")
                feats.append(to_tensor(self.model.get_image_features(**inp)).float().numpy())
        return l2(np.concatenate(feats).astype(np.float64))

    def text_embeddings(self, texts):
        if self.kind.startswith("siglip"):
            inp = self.processor(text=[t.lower() for t in texts], padding="max_length", max_length=64,
                                 truncation=True, return_tensors="pt")
        else:
            inp = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = to_tensor(self.model.get_text_features(**inp)).float().numpy()
        return l2(out.astype(np.float64))

    def score(self, cos):
        return 1.0 / (1.0 + np.exp(-(np.exp(self.scale) * cos + self.bias)))


def cache_path(cache_dir, name):
    return os.path.join(cache_dir, name.replace("/", "__") + ".npz")


def main():
    p = argparse.ArgumentParser(description="그림-문장 모델 후보 비교")
    p.add_argument("--manifest", required=True)
    p.add_argument("--answer-key", required=True)
    p.add_argument("--models", nargs="+", required=True, help="Hugging Face 모델 이름들")
    p.add_argument("--cache-dir", required=True, help="모델별 이미지 임베딩 캐시 폴더 (*.npz, 커밋 안 함)")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--top-k", type=int, required=True)
    p.add_argument("--min-pass", type=int, required=True, help="성공 기준 1의 통과 건수 (8)")
    p.add_argument("--min-remaining", type=int, required=True, help="구간 B의 남아야 할 정답 수 (8)")
    p.add_argument("--batch-size", type=int, required=True)
    a = p.parse_args()

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(a.results_dir, stamp + "_models")
    os.makedirs(out_dir)            # 이미 있으면 에러 → 덮어쓰지 않음
    os.makedirs(a.cache_dir, exist_ok=True)

    with open(a.manifest, newline="", encoding="utf-8") as f:
        items = sorted((r["asset_name"], r["png_path"]) for r in csv.DictReader(f))
    names = [n for n, _ in items]
    key = load_answer_key(a.answer_key)
    ans_q = [q for q in key if not q["absent"]]
    n_ans, n_abs = len(ans_q), len(key) - len(ans_q)

    rows, summary = [], []
    for mname in a.models:
        print("== " + mname)
        c = Candidate(mname)
        cp = cache_path(a.cache_dir, mname)
        t0 = time.time()
        if os.path.isfile(cp):
            d = np.load(cp)
            assert list(d["names"]) == names, "캐시의 에셋 목록이 manifest와 다름: " + cp
            img = d["embeddings"]
            img_sec = None
        else:
            img = c.image_embeddings([p for _, p in items], a.batch_size)
            img_sec = time.time() - t0
            np.savez_compressed(cp, embeddings=img, names=np.array(names), model=np.array(mname))
        idx = {n: i for i, n in enumerate(names)}

        for lang in ("ko", "en"):
            t1 = time.time()
            tx = c.text_embeddings([q["query_" + lang] for q in key])
            cos_all = tx @ img.T                         # (질의 수, 에셋 수)
            q_sec = (time.time() - t1) / len(key)
            graded = []
            for qi, q in enumerate(key):
                cos = cos_all[qi]
                sc = c.score(cos)
                ranked = sorted(names, key=lambda n: (-sc[idx[n]], n))
                g = grade(ranked, {n: float(sc[idx[n]]) for n in names}, q, a.top_k)
                g["absent"] = q["absent"]
                # 코사인 기준 동일 지표 (점수 형태와 무관한 비교용)
                gc = grade(ranked, {n: float(cos[idx[n]]) for n in names}, q, a.top_k)
                gc["absent"] = q["absent"]
                graded.append((g, gc))
                rows.append({"model": mname, "lang": lang, "id": q["id"], "query": q["query_" + lang],
                             "absent": int(q["absent"]), "pass": "" if g["pass"] is None else int(g["pass"]),
                             "first_answer_rank": g["first_answer_rank"] or "",
                             "first_answer_score": repr(g["first_answer_score"]) if g["first_answer_score"] is not None else "",
                             "traps_in_topk": g["traps_in_topk"], "top1_name": g["top1_name"],
                             "top1_score": repr(g["top1_score"]), "top1_cosine": repr(gc["top1_score"]),
                             "topk_names": g["topk_names"]})
            _, ex_s = sweep([g for g, _ in graded], [], n_ans, n_abs, a.min_remaining)
            _, ex_c = sweep([gc for _, gc in graded], [], n_ans, n_abs, a.min_remaining)
            ans_scores = [g["first_answer_score"] for g, _ in graded if not g["absent"] and g["first_answer_score"] is not None]
            summary.append({
                "model": mname, "lang": lang,
                "pass": sum(1 for g, _ in graded if g["pass"]),
                "traps": sum(g["traps_in_topk"] for g, _ in graded if not g["absent"]),
                "absent_max_score": ex_s["absent_max_top1"],
                "answer_mean_score": float(np.mean(ans_scores)) if ans_scores else None,
                "B_score": ex_s["B"], "B_cos": ex_c["B"], "A_score": ex_s["A"], "A_cos": ex_c["A"],
                "img_sec": img_sec, "query_sec": q_sec,
            })
        del c

    # 기준선(이름 검색, 영어만)
    base_pass, base_traps = 0, 0
    for q in ans_q:
        r, s = baseline_rank(q["query_en"], names)
        g = grade(r, s, q, a.top_k)
        base_pass += int(bool(g["pass"]))
        base_traps += g["traps_in_topk"]

    with open(os.path.join(out_dir, "per_query.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def iv(x):
        return "없음" if x is None else "{!r} < t ≤ {!r}".format(x[0], x[1])

    L = ["# 모델 후보 비교 ({})".format(stamp), "",
         "같은 썸네일 243장, 같은 정답표, 같은 채점 함수(evaluate.py). top-k={}, 성공 기준 1 = {}건 이상, 구간 B = 정답 {}건 이상 남음.".format(
             a.top_k, a.min_pass, a.min_remaining), "",
         "| 모델 | 질의 | top-{} 통과 | 함정 | absent 1위 점수 최대 | 정답 점수 평균 | 구간 B (점수) | 구간 B (코사인) | 이미지 243장 임베딩(초) | 질의 1건(초) |".format(a.top_k),
         "|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        L.append("| `{}` | {} | {}/{} | {} | {!r} | {!r} | {} | {} | {} | {} |".format(
            s["model"], "한국어" if s["lang"] == "ko" else "영어", s["pass"], n_ans, s["traps"],
            s["absent_max_score"], s["answer_mean_score"], iv(s["B_score"]), iv(s["B_cos"]),
            "캐시" if s["img_sec"] is None else repr(s["img_sec"]), repr(s["query_sec"])))
    L.append("| 이름 검색 (기준선) | 영어 | {}/{} | {} | — | — | — | — | — | — |".format(base_pass, n_ans, base_traps))
    L += ["", "점수 = sigmoid(exp(logit_scale) × 코사인 + logit_bias). CLIP은 logit_bias가 없어 0. "
          "구간 B (코사인)은 점수 대신 코사인에 임계값을 걸었을 때의 구간입니다. 시간은 이 PC의 CPU 기준 참고값입니다.",
          "", "질의별 상세: per_query.csv"]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("저장:", out_dir)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
