# -*- coding: utf-8 -*-
"""
2단계: 썸네일 → 이미지 임베딩 색인(index.npz) 만들기.

하는 일
  1. manifest.csv(1단계 결과)에서 에셋 이름과 PNG 경로를 읽습니다.
  2. SigLIP 2 모델로 각 PNG를 "이미지 임베딩"(숫자 벡터, 이 모델은 768개 숫자)으로 바꿉니다.
     - 임베딩: 그림의 의미를 숫자 목록으로 표현한 것. 비슷한 그림/문장은 비슷한 방향의 벡터가 됩니다.
  3. 각 벡터를 길이 1로 맞춥니다(L2 정규화). 그러면 나중에 '내적'이 곧 '코사인 유사도'가 됩니다.
  4. index.npz 한 파일에 저장합니다 (numpy 압축 파일, 벡터 DB 안 씀).

모든 계산은 로컬 CPU에서 합니다. 이미지는 어디로도 전송하지 않습니다.
(처음 실행할 때 모델 가중치만 Hugging Face에서 내려받아 캐시에 저장합니다.)

사용 예:
  .venv/Scripts/python build_index.py --manifest manifest.csv --out index.npz \
      --model google/siglip2-base-patch16-224 --batch-size 16
"""

import argparse
import csv
import datetime
import os
import sys

import numpy as np                      # 배열 계산·저장
import torch                            # 모델 실행 엔진 (CPU)
from PIL import Image                   # PNG 읽기
from transformers import AutoModel, AutoProcessor   # Hugging Face 모델/전처리기 자동 로더


def parse_args():
    p = argparse.ArgumentParser(description="썸네일 이미지 임베딩 색인 만들기")
    p.add_argument("--manifest", required=True, help="1단계 manifest.csv 경로")
    p.add_argument("--out", required=True, help="저장할 index.npz 경로 (이미 있으면 덮어쓰지 않고 중단)")
    p.add_argument("--model", required=True, help="Hugging Face 모델 이름")
    p.add_argument("--batch-size", type=int, required=True, help="한 번에 모델에 넣을 이미지 수")
    p.add_argument("--threads", type=int, default=0, help="CPU 스레드 수 (0 = torch 기본값)")
    return p.parse_args()


def load_manifest(path):
    """manifest.csv → [(에셋 이름, PNG 경로), ...] (이름순)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["asset_name"], r["png_path"]))
    rows.sort(key=lambda t: t[0])
    return rows


def to_tensor(out):
    """
    transformers 버전에 따라 get_image_features가 텐서를 바로 주거나,
    여러 값을 담은 출력 객체를 줍니다. 객체면 그 안의 pooler_output(그림 전체를 요약한 벡터)을 씁니다.
    """
    if isinstance(out, torch.Tensor):
        return out
    return out.pooler_output


def model_revision(model_name):
    """
    로컬 캐시에 받아 둔 모델 파일의 버전(커밋 해시)을 알아냅니다. (네트워크 안 씀)
    캐시 경로가 .../snapshots/<커밋 해시>/config.json 모양이라 그 폴더 이름을 읽습니다.
    """
    from huggingface_hub import try_to_load_from_cache
    path = try_to_load_from_cache(model_name, "config.json")
    if isinstance(path, str):
        return os.path.basename(os.path.dirname(path))
    return ""


def main():
    a = parse_args()

    # 중간 산출물은 덮어쓰지 않는다는 규칙 → 이미 있으면 바로 중단
    if os.path.exists(a.out):
        sys.exit("이미 있습니다: {} (지우거나 다른 --out 경로를 쓰세요)".format(a.out))

    if a.threads > 0:
        torch.set_num_threads(a.threads)

    items = load_manifest(a.manifest)
    missing = [p for _, p in items if not os.path.isfile(p)]
    if missing:
        sys.exit("PNG가 없습니다 ({}개). 예: {}".format(len(missing), missing[:3]))
    print("대상 {}개".format(len(items)))

    # 모델과 전처리기 불러오기. 전처리기는 그림 크기 조정(224×224)·정규화를 모델 학습 때와 똑같이 해 줍니다.
    print("모델 로드:", a.model)
    model = AutoModel.from_pretrained(a.model)
    processor = AutoProcessor.from_pretrained(a.model)
    model.eval()   # 추론 모드 (학습용 동작 끔)

    feats = []
    with torch.no_grad():   # 학습 안 하므로 기울기 계산 끔 → 빠르고 메모리 적게 씀
        for i in range(0, len(items), a.batch_size):
            batch = items[i: i + a.batch_size]
            # RGBA → RGB (알파는 전부 255라 버려도 그림은 같음)
            images = [Image.open(p).convert("RGB") for _, p in batch]
            inputs = processor(images=images, return_tensors="pt")
            emb = to_tensor(model.get_image_features(**inputs))   # (배치, 768)
            feats.append(emb.float().numpy())
            print("  {}/{}".format(min(i + a.batch_size, len(items)), len(items)))

    emb = np.concatenate(feats, axis=0).astype(np.float32)
    # L2 정규화: 각 행(벡터)을 자기 길이로 나눠 길이를 1로 만듦
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / norms

    names = np.array([n for n, _ in items])
    created = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    revision = model_revision(a.model)   # 받은 모델 파일의 버전(커밋 해시) → 재현용

    np.savez_compressed(
        a.out,
        embeddings=emb,            # (에셋 수, 차원) float32, 길이 1
        names=names,               # 에셋 이름 (embeddings와 같은 순서)
        model=np.array(a.model),   # 모델 이름
        model_revision=np.array(revision),
        created_at=np.array(created),
    )

    # 저장 확인용 요약
    lens = np.linalg.norm(emb, axis=1)
    print("저장: {}".format(a.out))
    print("  shape={} dtype={}".format(emb.shape, emb.dtype))
    print("  벡터 길이 최소/최대 = {!r} / {!r}".format(float(lens.min()), float(lens.max())))
    print("  model={} revision={} created_at={}".format(a.model, revision, created))


if __name__ == "__main__":
    main()
