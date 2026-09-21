#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def read(root, tag):
    with open(root / f"{tag}.json", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("phase", choices=("dcu", "batch", "final"))
    ap.add_argument("--dcu", type=int)
    ap.add_argument("--batch", type=int)
    args = ap.parse_args()

    if args.phase == "dcu":
        one = read(args.root, "fp32_dcu1_b2048")
        two = read(args.root, "fp32_dcu2_b1024")
        four = read(args.root, "fp32_dcu4_b512")
        ratio21 = two["samples_per_second"] / one["samples_per_second"]
        ratio42 = four["samples_per_second"] / two["samples_per_second"]
        selected = 1 if ratio21 < 1.5 else (4 if ratio42 >= 1.2 else 2)
        print(selected)
        return

    if args.phase == "batch":
        current_batch = 2048 // args.dcu
        doubled_batch = 4096 // args.dcu
        current = read(args.root, f"fp32_dcu{args.dcu}_b{current_batch}")
        doubled = read(args.root, f"fp32_dcu{args.dcu}_b{doubled_batch}")
        selected = doubled_batch if doubled["samples_per_second"] >= 1.05 * current["samples_per_second"] else current_batch
        print(selected)
        return

    rows = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in args.root.glob("*.json") if p.name != "selection.json"}
    one = rows["fp32_dcu1_b2048"]
    two = rows["fp32_dcu2_b1024"]
    four = rows["fp32_dcu4_b512"]
    ratio21 = two["samples_per_second"] / one["samples_per_second"]
    ratio42 = four["samples_per_second"] / two["samples_per_second"]
    fp32 = rows[f"fp32_dcu{args.dcu}_b{args.batch}"]
    amp = rows.get(f"fp16_dcu{args.dcu}_b{args.batch}")
    amp_ok = bool(amp and amp.get("loss_finite") and amp["samples_per_second"] >= 1.05 * fp32["samples_per_second"])
    result = {
        "dcu_count": args.dcu,
        "batch_size_per_rank": args.batch,
        "effective_batch_size": args.dcu * args.batch,
        "num_workers_total": 16,
        "amp_mode": "fp16" if amp_ok else "none",
        "scaling_2_over_1": ratio21,
        "scaling_4_over_2": ratio42,
        "amp_speedup": amp["samples_per_second"] / fp32["samples_per_second"] if amp else None,
        "selection_rules": {"2_over_1_min": 1.5, "4_over_2_min": 1.2, "batch_gain_min": 1.05, "amp_gain_min": 1.05},
        "runs": rows,
    }
    (args.root / "selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
