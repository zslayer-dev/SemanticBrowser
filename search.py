# -*- coding: utf-8 -*-
"""
3단계: 문장으로 에셋 찾기 (search.py)

하는 일
  1. index.npz(2단계)에서 이미지 임베딩(길이 1)과 에셋 이름을 읽습니다.
  2. 질의 문장을 '같은 모델'로 텍스트 임베딩한 뒤 길이 1로 맞춥니다.
  3. 코사인 유사도 = 이미지 벡터 · 텍스트 벡터 (둘 다 길이 1이라 내적이 곧 코사인)
  4. 점수 = sigmoid(exp(logit_scale) × 코사인 + logit_bias)
       - logit_scale, logit_bias는 모델이 학습한 값을 그대로 씁니다.
       - SigLIP은 "이 그림-문장 짝이 맞을 확률"을 각각 따로(sigmoid) 학습했으므로,
         점수는 0~1 사이의 독립적인 값입니다. 없는 것을 물으면 모든 점수가 낮게 나올 수 있습니다.
       - softmax(전체 합이 1이 되도록 나눔)는 쓰지 않습니다. softmax는 "다 아니다"를 표현할 수 없어서
         없는 것을 물어도 그중 하나가 높은 점수를 받기 때문입니다.
  5. 점수 높은 순으로 top-k개를 보여 줍니다. --threshold > 0이면 그 미만은 버립니다(결과가 비어도 됨).

텍스트 처리는 SigLIP 2 학습 조건을 따릅니다:
  - padding="max_length", max_length=64 (짧은 문장도 64칸을 채움)
  - 소문자 변환 (학습 기본값. --no-lowercase로 끌 수 있음. 한국어에는 영향 없음)

다른 스크립트(evaluate.py)에서도 쓸 수 있도록 Searcher 클래스로 묶었습니다.

사용 예:
  .venv/Scripts/python search.py --index index.npz --query "등불"
  .venv/Scripts/python search.py --index index.npz --query "pizza" --threshold 0.001
"""

import argparse
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor


class Searcher:
    """색인과 모델을 한 번만 불러 두고, 질의를 여러 번 처리할 수 있게 하는 객체."""

    def __init__(self, index_path, model_name=None, max_length=64, lowercase=True):
        d = np.load(index_path)
        self.image_emb = d["embeddings"]                      # (에셋 수, 차원), 길이 1
        self.names = [str(n) for n in d["names"]]             # 에셋 이름 (같은 순서)
        self.index_model = str(d["model"])                    # 색인을 만든 모델 이름

        # 색인과 다른 모델로 질의를 임베딩하면 벡터 공간이 달라 점수가 무의미해짐 → 막음
        if model_name and model_name != self.index_model:
            raise ValueError("색인 모델({})과 --model({})이 다릅니다".format(self.index_model, model_name))
        self.model_name = self.index_model
        self.max_length = max_length
        self.lowercase = lowercase

        self.model = AutoModel.from_pretrained(self.model_name)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model.eval()

        # 모델이 학습한 두 숫자. 점수 계산에 그대로 사용합니다.
        #   logit_scale: 코사인(-1~1)을 얼마나 늘릴지 (exp를 씌워서 씀)
        #   logit_bias : 늘린 값을 얼마나 옮길지 (음수가 크면 '기본적으로는 아니다' 쪽)
        self.logit_scale = float(self.model.logit_scale.detach())
        self.logit_bias = float(self.model.logit_bias.detach())

    def encode_text(self, texts):
        """문장 목록 → 길이 1로 맞춘 텍스트 임베딩 배열 (문장 수, 차원)."""
        if self.lowercase:
            texts = [t.lower() for t in texts]
        inputs = self.processor(text=texts, padding="max_length", max_length=self.max_length,
                                truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
        if not isinstance(out, torch.Tensor):   # transformers 버전에 따라 출력 객체일 수 있음
            out = out.pooler_output
        emb = out.float().numpy()
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    def scores(self, text):
        """
        문장 하나에 대한 모든 에셋의 (코사인, 점수) 배열을 돌려줍니다. 반올림하지 않습니다.
        점수 = sigmoid(exp(logit_scale) * 코사인 + logit_bias)
        """
        t = self.encode_text([text])[0]                     # (차원,)
        cos = self.image_emb.astype(np.float64) @ t.astype(np.float64)   # 각 에셋과의 코사인
        logits = np.exp(self.logit_scale) * cos + self.logit_bias
        score = 1.0 / (1.0 + np.exp(-logits))               # sigmoid
        return cos, score

    def search(self, text, top_k, threshold=0.0):
        """
        점수 높은 순으로 [(순위, 이름, 점수, 코사인), ...] 최대 top_k개.
        threshold > 0이면 점수가 threshold 미만인 것은 뺍니다(그래서 결과가 비어 있을 수 있음).
        동점이면 이름순으로 정렬합니다(실행마다 순서가 같도록).
        """
        cos, score = self.scores(text)
        order = sorted(range(len(self.names)), key=lambda i: (-score[i], self.names[i]))
        out = []
        for i in order[:top_k]:
            if threshold > 0 and score[i] < threshold:
                break
            out.append((len(out) + 1, self.names[i], float(score[i]), float(cos[i])))
        return out


def main():
    p = argparse.ArgumentParser(description="문장으로 에셋 검색 (SigLIP 2, sigmoid 점수)")
    p.add_argument("--index", required=True, help="2단계 index.npz 경로")
    p.add_argument("--query", required=True, help="찾을 문장 (한국어/영어)")
    p.add_argument("--top-k", type=int, default=5, help="보여 줄 개수 (기본 5)")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="이 점수 미만은 결과에서 뺌 (기본 0 = 사용 안 함)")
    p.add_argument("--model", default="", help="확인용: 색인 모델과 같아야 함 (비우면 색인의 모델 사용)")
    p.add_argument("--max-length", type=int, default=64, help="텍스트 토큰 길이 (SigLIP 2 학습 조건 64)")
    p.add_argument("--no-lowercase", action="store_true", help="질의를 소문자로 바꾸지 않음")
    a = p.parse_args()

    s = Searcher(a.index, a.model or None, a.max_length, not a.no_lowercase)
    res = s.search(a.query, a.top_k, a.threshold)

    print("질의: {}".format(a.query))
    print("모델: {}  logit_scale={!r} (exp={!r})  logit_bias={!r}".format(
        s.model_name, s.logit_scale, float(np.exp(s.logit_scale)), s.logit_bias))
    print("top-k={}  threshold={!r}".format(a.top_k, a.threshold))
    if not res:
        print("(결과 없음)")
        return
    print("순위\t에셋 이름\t점수\t코사인")
    for rank, name, score, cos in res:
        # repr: 반올림 없이 파이썬이 가진 그대로의 값을 출력
        print("{}\t{}\t{!r}\t{!r}".format(rank, name, score, cos))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")   # Windows 콘솔에서 한글이 깨지지 않게
    main()
