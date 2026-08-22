#!/usr/bin/env python3
"""Aggregate fresh-reload DL3DV host/compact RGB and storage results."""

import argparse
import json
from pathlib import Path


SCENES = ("supermarket", "furniture_store", "museum", "business_center")


def load(path):
    with path.open() as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="/data2/jian/outputs/wacv27_dl3dv_20260819"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    rows = {}
    missing = []
    for scene in SCENES:
        host = root / scene / "semantic_host"
        ours = root / scene / "clunogs"
        paths = {
            "host_rgb": host / "evaluation" / "rgb.json",
            "ours_rgb": ours / "evaluation" / "rgb.json",
            "host_manifest": host / "deploy" / (scene + ".host.pth.manifest.json"),
            "ours_manifest": ours / "deploy" / (scene + ".ours.pth.manifest.json"),
        }
        absent = [name for name, path in paths.items() if not path.is_file()]
        if absent:
            missing.append({"scene": scene, "files": absent})
            continue
        host_rgb = load(paths["host_rgb"])
        ours_rgb = load(paths["ours_rgb"])
        host_manifest = load(paths["host_manifest"])
        ours_manifest = load(paths["ours_manifest"])
        host_bytes = int(host_manifest["one_scene_total_bytes"])
        ours_bytes = int(ours_manifest["one_scene_total_bytes"])
        rows[scene] = {
            "source_point_count": int(host_manifest["point_count"]),
            "compact_point_count": int(ours_manifest["point_count"]),
            "host_psnr": float(host_rgb["metrics"]["psnr"]),
            "ours_psnr": float(ours_rgb["metrics"]["psnr"]),
            "host_storage_mb_decimal": host_bytes / 1e6,
            "ours_storage_mb_decimal": ours_bytes / 1e6,
            "compression_ratio": host_bytes / ours_bytes,
        }

    keys = (
        "source_point_count",
        "compact_point_count",
        "host_psnr",
        "ours_psnr",
        "host_storage_mb_decimal",
        "ours_storage_mb_decimal",
        "compression_ratio",
    )
    macro = (
        {key: sum(row[key] for row in rows.values()) / len(rows) for key in keys}
        if rows
        else {}
    )
    result = {
        "protocol": "fresh-reload RGB; complete artifact; decimal MB; scene macro",
        "per_scene": rows,
        "macro": macro,
        "missing": missing,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
