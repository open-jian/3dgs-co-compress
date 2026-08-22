#!/usr/bin/env python3
"""Summarize matched source-ID support changes for semantic guidance."""

import argparse
import json
from pathlib import Path

import torch


SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")


def load_ids(path):
    payload = torch.load(str(path), map_location="cpu")
    ids = payload["remaining_source_ids"].long().reshape(-1)
    return payload, set(ids.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="/data2/jian/outputs/wacv27_section6_20260819"
    )
    parser.add_argument("--iteration", type=int, default=1000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    rows = {}
    missing = []
    for scene in SCENES:
        full_path = root / "full_tracked" / scene / (
            "chkpnt%d.pth.source_ids.pt" % args.iteration
        )
        nosem_path = root / "no_sem_tracked" / scene / (
            "chkpnt%d.pth.source_ids.pt" % args.iteration
        )
        if not full_path.is_file() or not nosem_path.is_file():
            missing.append(scene)
            continue
        full_meta, full = load_ids(full_path)
        nosem_meta, nosem = load_ids(nosem_path)
        if full_meta["origin_xyz_sha256"] != nosem_meta["origin_xyz_sha256"]:
            raise RuntimeError("origin mismatch for %s" % scene)
        inter = full & nosem
        union = full | nosem
        only_full = full - nosem
        rows[scene] = {
            "origin_point_count": int(full_meta["origin_point_count"]),
            "full_count": len(full),
            "without_semantic_count": len(nosem),
            "intersection_count": len(inter),
            "semantic_only_count": len(only_full),
            "retained_overlap_percent": 100.0 * len(inter) / max(len(full), 1),
            "semantic_only_percent_of_retained": 100.0 * len(only_full) / max(len(full), 1),
            "jaccard_percent": 100.0 * len(inter) / max(len(union), 1),
        }

    metrics = (
        "retained_overlap_percent",
        "semantic_only_percent_of_retained",
        "jaccard_percent",
    )
    macro = (
        {
            key: sum(row[key] for row in rows.values()) / len(rows)
            for key in metrics
        }
        if rows
        else {}
    )
    result = {
        "protocol": "matched source-row IDs; scene-macro percentages",
        "per_scene": rows,
        "macro": macro,
        "missing_scenes": missing,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
