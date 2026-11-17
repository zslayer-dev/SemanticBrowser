# -*- coding: utf-8 -*-
"""
썸네일 PNG를 '숫자로' 점검하는 스크립트 (일반 Python에서 실행, 이미지를 외부로 보내지 않음).

각 PNG마다:
  - 크기(가로×세로), 색 모드(RGB/RGBA), 알파(투명도) 최소값
  - 배경색: 네 모서리 픽셀의 중앙값으로 추정
  - 물체 비율: 배경색과 tol 이상 다른 픽셀의 비율 (0이면 물체가 안 찍힌 것)
  - 가장자리 닿음: 물체 픽셀이 이미지 테두리에 닿았는지 (잘렸을 가능성)
결과를 CSV로 저장하고 요약을 출력합니다.

사용 예:
  python tools/check_thumbs.py --dir thumbs --out thumbs/_check.csv --tol 12
"""
import argparse
import csv
import glob
import os

import numpy as np            # 배열 계산
from PIL import Image         # PNG 읽기


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="PNG 폴더")
    p.add_argument("--out", required=True, help="점검 결과 CSV 경로")
    p.add_argument("--tol", type=int, required=True, help="배경과 '다르다'고 볼 채널 차이(0~255)")
    a = p.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(a.dir, "*.png"))):
        im = Image.open(path)
        mode = im.mode
        arr = np.asarray(im.convert("RGBA")).astype(np.int16)   # (세로, 가로, 4)
        h, w, _ = arr.shape
        alpha_min = int(arr[..., 3].min())
        rgb = arr[..., :3]
        corners = np.stack([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]])
        bg = np.median(corners, axis=0)                          # 배경색 추정
        diff = np.abs(rgb - bg).max(axis=2)                      # 픽셀별 배경과의 최대 채널 차이
        obj = diff > a.tol                                       # True = 물체 픽셀
        frac = float(obj.mean())
        edge = bool(obj[0].any() or obj[-1].any() or obj[:, 0].any() or obj[:, -1].any())
        rows.append({
            "file": os.path.basename(path), "w": w, "h": h, "mode": mode, "alpha_min": alpha_min,
            "bg_r": bg[0], "bg_g": bg[1], "bg_b": bg[2],
            "object_fraction": frac, "touches_edge": edge,
            "mean": float(rgb.mean()), "std": float(rgb.std()),
        })

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["file"])
        wr.writeheader()
        wr.writerows(rows)

    print("PNG {}개".format(len(rows)))
    if rows:
        fr = np.array([r["object_fraction"] for r in rows])
        print("크기 종류:", sorted({(r["w"], r["h"], r["mode"]) for r in rows}))
        print("알파 최소값 종류:", sorted({r["alpha_min"] for r in rows}))
        print("배경색 종류:", sorted({(r["bg_r"], r["bg_g"], r["bg_b"]) for r in rows}))
        print("물체 비율 최소/중앙/최대: {} / {} / {}".format(fr.min(), np.median(fr), fr.max()))
        print("물체 비율 < 1%:", [r["file"] for r in rows if r["object_fraction"] < 0.01])
        print("가장자리 닿음:", [r["file"] for r in rows if r["touches_edge"]])


if __name__ == "__main__":
    main()
