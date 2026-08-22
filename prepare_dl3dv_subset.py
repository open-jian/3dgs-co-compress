#!/usr/bin/env python3
"""Create a deterministic, uniformly sampled COLMAP-view subset."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


SCENES = ("supermarket", "furniture_store", "museum", "business_center")


def evenly_spaced_indices(length, count):
    if count > length:
        raise ValueError("requested %d views from only %d" % (count, length))
    indices = np.linspace(0, length - 1, num=count, dtype=np.int64)
    if len(set(indices.tolist())) != count:
        raise RuntimeError("uniform sampling produced duplicate indices")
    return indices.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--colmap-utils", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.colmap_utils)
    from read_write_model import (  # pylint: disable=import-error,import-outside-toplevel
        read_images_binary,
        write_images_binary,
    )

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"count": args.count, "sampling": "uniform over sorted image names", "scenes": {}}

    for scene in SCENES:
        source = source_root / scene
        output = output_root / scene
        images = read_images_binary(str(source / "sparse" / "images.bin"))
        ordered = sorted(images.values(), key=lambda item: item.name)
        selected = [ordered[index] for index in evenly_spaced_indices(len(ordered), args.count)]
        selected_by_id = {item.id: item for item in selected}

        image_dir = output / "images"
        sparse_dir = output / "sparse" / "0"
        image_dir.mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        for item in selected:
            source_image = source / "images" / item.name
            target_image = image_dir / item.name
            if not target_image.exists():
                os.symlink(str(source_image), str(target_image))

        for filename in ("cameras.bin", "points3D.bin"):
            target = sparse_dir / filename
            if not target.exists():
                os.symlink(str(source / "sparse" / filename), str(target))
        write_images_binary(selected_by_id, str(sparse_dir / "images.bin"))

        names = [item.name for item in selected]
        (output / "selected_images.txt").write_text("\n".join(names) + "\n")
        manifest["scenes"][scene] = {
            "source_view_count": len(ordered),
            "selected_view_count": len(selected),
            "selected_images": names,
        }

    (output_root / "subset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
