#!/usr/bin/env python3
"""Regression 檢查: 用 tests/regression_cases.json 的片段重跑落點分析,
和使用者肉眼確認過的案例逐一比對。

第一階段 (球/桌面偵測) 讀 data/cache/ 的快取 (第一次跑會自動建立),
只重跑第二階段 (事件 + 落點規則), 整組幾秒鐘。改擊球偵測或規則後跑這支即可。

結果標記:
  ✅ 通過   ❌ 變了 (原本應通過, 現在沒有)
  ⚠️ 已知問題仍在   🎯 目標尚未修好
  🎉 已知問題/目標現在通過了 (可把 status 改成 pass)

用法:
    python scripts/regression.py                 # 全部
    python scripts/regression.py --clip lin_0217 # 只跑某片段
    python scripts/regression.py --save-dir data/regression   # 順便輸出各片段 JSON (可拿去做 demo)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.landing_points import (analyze, build_parser, load_or_run_detectors,  # noqa: E402
                                    load_prof_module, patch_debounce)

CASES = PROJECT_ROOT / "tests" / "regression_cases.json"
DEFAULT_TOL = 0.1   # s


def is_counted(e: dict) -> bool:
    """在桌上且算數 (發球第 1 跳不計數, 與小視窗一致)。"""
    return bool(e.get("zone")) and not e.get("uncounted") and e.get("serve") != 1


def label(e: dict) -> str:
    if e["type"] == "hit":
        return f"擊球 {e['t']:.2f}"
    s = f"落點 {e['t']:.2f} {e.get('zone') or '桌外'}"
    if e.get("serve"):
        s += f" SERVE {e['serve']}"
    if e.get("uncounted"):
        s += f" ({e['uncounted']})"
    return s


def check(case: dict, events: list[dict]) -> tuple[bool, str]:
    """回傳 (是否通過, 實際結果的描述)。"""
    t, tol, exp = case["t"], case.get("tol", DEFAULT_TOL), case["expect"]
    bounces = [e for e in events if e["type"] == "bounce"]
    hits = [e for e in events if e["type"] == "hit"]

    def near(evs, t0=t, tl=tol):
        c = [e for e in evs if abs(e["t"] - t0) <= tl + 1e-6]
        return min(c, key=lambda e: abs(e["t"] - t0)) if c else None

    if exp == "none_counted_in":
        bad = [e for e in bounces if t - tol <= e["t"] <= case["t_end"] + tol and is_counted(e)]
        return not bad, ("沒有算數的落點" if not bad else ", ".join(label(e) for e in bad))
    if exp in ("hit", "no_hit"):
        h = near(hits)
        ok = (h is not None) == (exp == "hit")
        b = near(bounces)
        return ok, (label(h) if h else "沒有擊球") + (f"; 附近 {label(b)}" if b else "")

    b = near(bounces)
    got = label(b) if b else "沒有落點"
    if exp == "no_bounce":
        return b is None, got
    if exp == "bounce":
        return b is not None, got
    if exp in ("serve1", "serve2"):
        return b is not None and b.get("serve") == int(exp[-1]), got
    if exp == "counted":
        ok = b is not None and is_counted(b)
        if ok and case.get("side"):
            ok = b["zone"][0] == case["side"]
        return ok, got
    if exp == "not_counted":
        return b is None or not is_counted(b), got
    raise ValueError(f"未知的 expect: {exp}")


def fmt_t(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", action="append", help="只跑這些片段 (可重複)")
    ap.add_argument("--refresh-cache", action="store_true", help="重跑第一階段 (球/桌面偵測)")
    ap.add_argument("--save-dir", default=None, help="把各片段的落點 JSON 寫到這裡 (供 render_landing_demo.py)")
    opts = ap.parse_args()

    spec = json.loads(CASES.read_text())
    clips = {k: v for k, v in spec["clips"].items() if not opts.clip or k in opts.clip}
    parser = build_parser()

    # 教授模組只載入/修補一次 (import 有快取, 重複 patch_debounce 會一層層包起來)
    defaults = parser.parse_args(["_"])
    de = load_prof_module(Path(defaults.prof_repo))
    if not defaults.no_fix_debounce:
        patch_debounce(de)
    min_speed0 = (de.MIN_Y_SPEED, de.MIN_X_SPEED)

    rows = []
    for name, c in clips.items():
        argv = [str(PROJECT_ROOT / c["video"]), "--start", str(c["start"]), "--dur", str(c["dur"])]
        if opts.refresh_cache:
            argv.append("--refresh-cache")
        args = parser.parse_args(argv)
        # analyze 會依桌面大小改 de.MIN_Y_SPEED / MIN_X_SPEED; 每個片段從教授原值開始, 不受上一個片段影響
        de.MIN_Y_SPEED, de.MIN_X_SPEED = min_speed0
        result, _ = analyze(load_or_run_detectors(args), args, de)
        if opts.save_dir:
            out = Path(opts.save_dir) / f"landing_{name}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        for case in spec["cases"]:
            if case["clip"] == name:
                ok, got = check(case, result["events"])
                rows.append((case, ok, got))

    counts = {"✅": 0, "❌": 0, "⚠️": 0, "🎯": 0, "🎉": 0}
    print()
    for case, ok, got in rows:
        st = case.get("status", "pass")
        mark = ("✅" if ok else "❌") if st == "pass" else ("🎉" if ok else ("⚠️" if st == "known_issue" else "🎯"))
        counts[mark] += 1
        exp = case["expect"] + (f" {case['side']}" if case.get("side") else "") + \
            (f" ~{fmt_t(case['t_end'])}" if case.get("t_end") else "")
        print(f"{mark} {case['clip']:12s} {fmt_t(case['t'])}  預期 {exp:18s} 實際 {got}")
    print("\n" + "   ".join(f"{k} {v}" for k, v in counts.items()))
    sys.exit(1 if counts["❌"] else 0)


if __name__ == "__main__":
    main()
