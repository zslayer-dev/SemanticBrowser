# -*- coding: utf-8 -*-
"""
계산 로직 단위 테스트 (모델·썸네일·색인 없이 돌아감).

실행:  .venv/Scripts/python -m unittest discover -s tests -v

검사하는 것
  - 정답표가 확정 커밋(095c745) 이후 바뀌지 않았는지 (sha256 비교)
  - search.py: 점수 = sigmoid(exp(logit_scale) * 코사인 + logit_bias), 정렬·동점·임계값
  - evaluate.py: 기준선 이름 검색, 채점(grade), 임계값 구간(sweep)
  - serve.py: manifest → 에셋 경로 변환
"""

import csv
import hashlib
import math
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluate import baseline_rank, grade, grid_range, sweep   # noqa: E402
from search import Searcher                                    # noqa: E402
from serve import load_paths                                   # noqa: E402

# 095c745 커밋의 answer_key.csv sha256 (모델 실행 전에 확정한 정답표)
ANSWER_KEY_SHA256 = "0b7f0b81eff3c8ac132641f06dc9369215c6175886f687eb5d1b5d20cc75c8a8"


def fake_searcher(image_emb, names, text_vec, logit_scale, logit_bias):
    """모델을 불러오지 않는 가짜 Searcher. 텍스트 임베딩은 고정 벡터를 돌려줌."""
    s = object.__new__(Searcher)   # __init__(모델 로드)을 건너뜀
    s.image_emb = np.asarray(image_emb, dtype=np.float32)
    s.names = list(names)
    s.logit_scale = logit_scale
    s.logit_bias = logit_bias
    v = np.asarray(text_vec, dtype=np.float32)
    s.encode_text = lambda texts: np.stack([v / np.linalg.norm(v)] * len(texts))
    return s


def q(answers=(), traps=(), absent=False):
    return {"answers": list(answers), "traps": list(traps), "absent": absent}


class TestAnswerKey(unittest.TestCase):
    def test_answer_key_unchanged(self):
        with open(os.path.join(ROOT, "answer_key.csv"), "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), ANSWER_KEY_SHA256)

    def test_answer_key_shape(self):
        with open(os.path.join(ROOT, "answer_key.csv"), newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 13)
        self.assertEqual(sum(r["absent"] == "1" for r in rows), 3)
        for r in rows:   # absent면 정답이 없고, 아니면 정답이 있어야 함
            self.assertEqual(r["absent"] == "1", r["answers"] == "")


class TestSearchScore(unittest.TestCase):
    def setUp(self):
        # 이미지 3개: 텍스트(1,0)과의 코사인이 각각 1, 0, 0
        self.s = fake_searcher([[1, 0], [0, 1], [0, 1]], ["B", "C", "A"], [1, 0],
                               logit_scale=math.log(10.0), logit_bias=-5.0)

    def test_score_formula_is_sigmoid_with_learned_scale_and_bias(self):
        cos, score = self.s.scores("x")
        np.testing.assert_allclose(cos, [1.0, 0.0, 0.0], atol=1e-7)
        expected = [1 / (1 + math.exp(-(10 * c - 5))) for c in (1.0, 0.0, 0.0)]
        np.testing.assert_allclose(score, expected, rtol=1e-12)

    def test_scores_are_independent_not_softmax(self):
        # softmax라면 합이 1. sigmoid는 독립이라 합이 1이 아님
        _, score = self.s.scores("x")
        self.assertNotAlmostEqual(float(score.sum()), 1.0, places=3)

    def test_ranking_ties_broken_by_name(self):
        res = self.s.search("x", top_k=3)
        self.assertEqual([n for _, n, _, _ in res], ["B", "A", "C"])   # A, C 동점 → 이름순
        self.assertEqual([r for r, _, _, _ in res], [1, 2, 3])

    def test_threshold_can_make_result_empty(self):
        self.assertEqual(self.s.search("x", top_k=3, threshold=0.999), [])
        self.assertEqual(len(self.s.search("x", top_k=3, threshold=0.0)), 3)   # 0 = 사용 안 함

    def test_scores_not_rounded(self):
        _, score = self.s.scores("x")
        self.assertEqual(float(score[0]), 1 / (1 + math.exp(-5.0)))


class TestBaseline(unittest.TestCase):
    NAMES = ["SM_wooden_fence_02", "SM_wooden_fence_01", "SM_fence_01", "SM_wall_arch_01", "SM_lantern_01"]

    def test_overlap_order_and_ties(self):
        ranked, scored = baseline_rank("Wooden fence", self.NAMES)
        self.assertEqual(ranked, ["SM_wooden_fence_01", "SM_wooden_fence_02", "SM_fence_01"])
        self.assertEqual(scored["SM_wooden_fence_01"], 2)
        self.assertEqual(scored["SM_fence_01"], 1)

    def test_no_overlap_means_no_result(self):
        self.assertEqual(baseline_rank("arched gateway", self.NAMES), ([], {}))   # arched ≠ arch
        self.assertEqual(baseline_rank("등불", self.NAMES), ([], {}))


class TestGrade(unittest.TestCase):
    def test_pass_first_answer_and_traps(self):
        ranked = ["T1", "X", "A2", "T2", "Y", "A1"]
        scores = {n: 1.0 / (i + 1) for i, n in enumerate(ranked)}
        g = grade(ranked, scores, q(["A1", "A2"], ["T1", "T2"]), top_k=5)
        self.assertTrue(g["pass"])
        self.assertEqual((g["first_answer_rank"], g["first_answer_name"]), (3, "A2"))
        self.assertEqual(g["first_answer_score"], scores["A2"])
        self.assertEqual(g["traps_in_topk"], 2)
        self.assertEqual(g["best_answer_score_in_topk"], scores["A2"])   # A1은 6위라 제외

    def test_fail_when_answer_outside_topk(self):
        ranked = ["X1", "X2", "A"]
        g = grade(ranked, {n: 1.0 for n in ranked}, q(["A"]), top_k=2)
        self.assertFalse(g["pass"])
        self.assertEqual(g["first_answer_rank"], 3)
        self.assertIsNone(g["best_answer_score_in_topk"])

    def test_absent_query_has_no_pass_value(self):
        g = grade(["X"], {"X": 0.1}, q(absent=True), top_k=5)
        self.assertIsNone(g["pass"])
        self.assertEqual((g["top1_name"], g["top1_score"]), ("X", 0.1))

    def test_empty_baseline_result(self):
        g = grade([], {}, q(["A"]), top_k=5)
        self.assertFalse(g["pass"])
        self.assertIsNone(g["first_answer_rank"])
        self.assertEqual(g["result_count"], 0)


class TestSweep(unittest.TestCase):
    @staticmethod
    def per_query(answer_best, absent_top1):
        rows = [{"absent": False, "best_answer_score_in_topk": s, "top1_score": None} for s in answer_best]
        rows += [{"absent": True, "best_answer_score_in_topk": None, "top1_score": s} for s in absent_top1]
        return rows

    def test_exact_bounds_A_and_B(self):
        pq = self.per_query([0.9, 0.5, 0.3, 0.2], [0.1, 0.05])
        rows, ex = sweep(pq, [0.05, 0.15, 0.25, 0.4], n_answer_q=4, n_absent_q=2, min_remaining=3)
        self.assertEqual(ex["absent_max_top1"], 0.1)
        self.assertEqual(ex["A"], (0.1, 0.2))   # 0.1 < t ≤ 0.2 에서 4건 모두 남음
        self.assertEqual(ex["B"], (0.1, 0.3))   # 3건 이상
        self.assertEqual([r["A"] for r in rows], [False, True, False, False])
        self.assertEqual([r["B"] for r in rows], [False, True, True, False])
        self.assertEqual(grid_range(rows, "B"), (0.15, 0.25, 2))

    def test_absent_empty_needs_strictly_greater(self):
        # 1위 점수 < t 일 때만 '빔' → t == absent 점수면 아직 안 빔
        pq = self.per_query([0.9], [0.1])
        rows, _ = sweep(pq, [0.1], n_answer_q=1, n_absent_q=1, min_remaining=1)
        self.assertEqual(rows[0]["absent_empty"], 0)

    def test_no_interval_reported_as_none(self):
        # 정답 점수가 absent보다 낮으면 구간 없음
        pq = self.per_query([0.05, 0.01], [0.1])
        rows, ex = sweep(pq, [0.001, 0.01, 0.1, 1.0], n_answer_q=2, n_absent_q=1, min_remaining=1)
        self.assertIsNone(ex["A"])
        self.assertIsNone(ex["B"])
        self.assertIsNone(grid_range(rows, "A"))

    def test_answer_missing_from_topk_never_remains(self):
        pq = self.per_query([0.9, None], [0.1])
        _, ex = sweep(pq, [0.5], n_answer_q=2, n_absent_q=1, min_remaining=1)
        self.assertIsNone(ex["A"])
        self.assertEqual(ex["B"], (0.1, 0.9))


class TestServePaths(unittest.TestCase):
    def test_load_paths(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.csv")
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["asset_name", "asset_path", "png_path", "note"])
                w.writerow(["SM_a", "/Game/X/SM_a", "x.png", "stable"])
            self.assertEqual(load_paths(p), {"SM_a": ("/Game/X/SM_a", "/Game/X/SM_a.SM_a")})


if __name__ == "__main__":
    unittest.main()
