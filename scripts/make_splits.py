#!/usr/bin/env python3
"""場地感知的 train/val 切分。

涵蓋 6 個場地 (由檔名前綴判斷)。每個場地內按影格序號排序,取後 --val-frac
當 val (場地內時序隔離,避免相鄰影格洩漏;所有場地都出現在 train)。
輸出 train.txt / val.txt,含正樣本 (有標註) 與負樣本 (無桌面特寫)。

用法:
    python scripts/make_splits.py [--val-frac 0.25]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

# 場地判斷: (檔名子字串, 場地名),依序比對
VENUE_RULES = [
    ("20230418", "Macau"),
    ("20231102", "Frankfurt"),
    ("20231105", "Frankfurt"),
    ("20241105", "Frankfurt"),
    ("20250312Chongqing", "Chongqing"),
    ("20250523World", "Doha"),
    ("20250712USA", "Vegas"),
    ("Incheon", "Incheon"),
]


def venue_of(stem: str) -> str:
    for key, name in VENUE_RULES:
        if key in stem:
            return name
    return "Unknown"


def frame_idx(stem: str) -> int:
    tail = stem.rsplit("_f", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-frac", type=float, default=0.25)
    args = parser.parse_args()

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    annotated = {p.stem for p in ANNOTATIONS_DIR.glob("*.json")}

    by_venue: dict[str, list[str]] = defaultdict(list)
    for frame in FRAMES_DIR.glob("*.jpg"):
        by_venue[venue_of(frame.stem)].append(frame.stem)

    train, val = [], []
    print(f"{'場地':10s} {'總':>4s} {'標註':>4s} {'train':>6s} {'val':>4s}")
    for venue, stems in sorted(by_venue.items()):
        stems.sort(key=lambda s: (s.rsplit("_f", 1)[0], frame_idx(s)))  # 同片段相鄰
        n_val = max(1, round(len(stems) * args.val_frac))
        v_stems, t_stems = stems[-n_val:], stems[:-n_val]
        train.extend(t_stems)
        val.extend(v_stems)
        n_ann = sum(s in annotated for s in stems)
        print(f"{venue:10s} {len(stems):4d} {n_ann:4d} {len(t_stems):6d} {len(v_stems):4d}")

    (SPLITS_DIR / "train.txt").write_text("\n".join(sorted(train)) + "\n")
    (SPLITS_DIR / "val.txt").write_text("\n".join(sorted(val)) + "\n")
    print(f"\ntrain {len(train)} / val {len(val)} 張,輸出於 {SPLITS_DIR}")


if __name__ == "__main__":
    main()
