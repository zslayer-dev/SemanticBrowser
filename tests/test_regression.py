# -*- coding: utf-8 -*-
"""
회귀 테스트: 커밋된 채점 결과(results/20261117_171146/per_query.csv)가 지금 코드로 그대로 재현되는지.

index.npz는 에셋 산출물이라 커밋하지 않으므로, 없으면 이 테스트는 건너뜁니다(skip).
처음 실행하면 모델 가중치를 Hugging Face에서 받습니다(로컬 캐시에 있으면 받지 않음).

실행:  .venv/Scripts/python -m unittest discover -s tests -v
"""

import csv
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

INDEX = os.path.join(ROOT, "index.npz")
PER_QUERY = os.path.join(ROOT, "results", "20261117_171146", "per_query.csv")
ANSWER_KEY = os.path.join(ROOT, "answer_key.csv")
REL_TOL = 1e-6   # CPU 스레드 수 등에 따른 부동소수점 미세 차이만 허용


@unittest.skipUnless(os.path.isfile(INDEX), "index.npz 없음 (에셋 산출물이라 커밋 안 됨) → 건너뜀")
class TestReproduceCommittedResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from evaluate import grade, load_answer_key
        from search import Searcher
        cls.grade = staticmethod(grade)
        cls.key = {r["id"]: r for r in load_answer_key(ANSWER_KEY)}
        cls.s = Searcher(INDEX)
        with open(PER_QUERY, newline="", encoding="utf-8") as f:
            cls.rows = [r for r in csv.DictReader(f) if r["method"] == "siglip2"]

    def _now(self, qid, lang):
        q = self.key[qid]
        _, sc = self.s.scores(q["query_" + lang])
        idx = {n: i for i, n in enumerate(self.s.names)}
        ranked = sorted(self.s.names, key=lambda n: (-sc[idx[n]], n))
        return self.grade(ranked, {n: float(sc[i]) for i, n in enumerate(self.s.names)}, q, 5)

    def test_all_queries_reproduce(self):
        self.assertEqual(len(self.rows), 26)   # 13건 × 한국어/영어
        for r in self.rows:
            with self.subTest(id=r["id"], lang=r["lang"]):
                g = self._now(r["id"], r["lang"])
                self.assertEqual(g["topk_names"], r["topk_names"])
                self.assertEqual(g["traps_in_topk"], int(r["traps_in_topk"]))
                if r["first_answer_rank"]:
                    self.assertEqual(g["first_answer_rank"], int(r["first_answer_rank"]))
                    self.assertAlmostEqual(g["first_answer_score"] / float(r["first_answer_score"]), 1.0,
                                           delta=REL_TOL)
                self.assertAlmostEqual(g["top1_score"] / float(r["top1_score"]), 1.0, delta=REL_TOL)


if __name__ == "__main__":
    unittest.main()
