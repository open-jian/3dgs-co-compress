#!/usr/bin/env python3
"""Create a deterministic 112-frame ScanNet mini scene from public RGB-D files."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image


def frame_id(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        raise ValueError(f"Cannot parse frame id from {path.name}")
    return int(match.group(1))


def read_info(path: Path):
    result = {}
    for line in path.read_text().splitlines():
        if " = " not in line:
            continue
        key, value = line.split(" = ", 1)
        result[key.strip()] = value.strip()
    return result


def safe_link(source: Path, destination: Path):
    source = source.resolve()
    if destination.is_symlink():
        if destination.resolve() != source:
            raise FileExistsError(f"Conflicting symlink: {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.symlink_to(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--gt_ply", type=Path, required=True)
    parser.add_argument("--scene_info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame_count", type=int, default=112)
    args = parser.parse_args()

    modalities = {}
    for name in ("color", "depth", "pose"):
        entries = {frame_id(path): path for path in (args.source / name).iterdir() if path.is_file()}
        modalities[name] = entries
    common = sorted(set.intersection(*(set(value) for value in modalities.values())))
    valid = []
    invalid_pose = []
    for identifier in common:
        pose = np.loadtxt(modalities["pose"][identifier])
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or abs(np.linalg.det(pose)) < 1e-10:
            invalid_pose.append(identifier)
        else:
            valid.append(identifier)
    if len(valid) < args.frame_count:
        raise ValueError(f"Only {len(valid)} valid matched frames, need {args.frame_count}")
    positions = np.linspace(0, len(valid) - 1, args.frame_count).round().astype(np.int64)
    selected = [valid[int(index)] for index in positions]
    if len(set(selected)) != args.frame_count:
        raise AssertionError("Uniform sampling produced duplicate frames")

    for name in ("color", "depth", "pose", "intrinsic"):
        (args.output / name).mkdir(parents=True, exist_ok=True)
    for identifier in selected:
        for name in ("color", "depth", "pose"):
            source = modalities[name][identifier]
            safe_link(source, args.output / name / f"{identifier}{source.suffix.lower()}")
    images_link = args.output / "images"
    if not images_link.exists() and not images_link.is_symlink():
        images_link.symlink_to("color")
    safe_link(args.gt_ply, args.output / "points3d.ply")
    safe_link(args.gt_ply, args.output / args.gt_ply.name)

    info = read_info(args.scene_info)
    first_image = Image.open(modalities["color"][selected[0]])
    image_width, image_height = first_image.size
    source_width = float(info["colorWidth"])
    source_height = float(info["colorHeight"])
    intrinsic_path = args.source / "intrinsic" / "intrinsic_color.txt"
    if intrinsic_path.exists():
        intrinsic = np.loadtxt(intrinsic_path).astype(np.float64)
    else:
        intrinsic = np.eye(4, dtype=np.float64)
        intrinsic[0, 0] = float(info["fx_color"])
        intrinsic[1, 1] = float(info["fy_color"])
        intrinsic[0, 2] = float(info["mx_color"])
        intrinsic[1, 2] = float(info["my_color"])
    intrinsic[0, :] *= image_width / source_width
    intrinsic[1, :] *= image_height / source_height
    intrinsic[2, 2] = 1.0
    intrinsic[3, 3] = 1.0
    output_intrinsic = args.output / "intrinsic" / "intrinsic_color.txt"
    if output_intrinsic.exists():
        if not np.allclose(np.loadtxt(output_intrinsic), intrinsic):
            raise FileExistsError(f"Conflicting intrinsic: {output_intrinsic}")
    else:
        np.savetxt(output_intrinsic, intrinsic, fmt="%.9f")

    manifest = {
        "schema": "scannet_public_uniform_mini_v1",
        "source": str(args.source.resolve()),
        "scene_info": str(args.scene_info.resolve()),
        "gt_ply": str(args.gt_ply.resolve()),
        "matched_frame_count": len(common),
        "valid_pose_frame_count": len(valid),
        "invalid_pose_frame_ids": invalid_pose,
        "selected_frame_count": len(selected),
        "selected_frame_ids": selected,
        "selection": "round(linspace(0, valid_count-1, frame_count)) over sorted valid matched ids",
        "split": "every eighth selected mini frame held out: 98 train / 14 test",
        "image_size": [image_width, image_height],
        "scaled_intrinsic": intrinsic.tolist(),
    }
    manifest_path = args.output / "mini_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
