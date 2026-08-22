#!/usr/bin/env python3
"""Aggregate Section 6 controls under one common LERF protocol."""

import argparse
import glob
import json
import os
from statistics import mean


SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")


def load(path):
    with open(path, "r") as handle:
        return json.load(handle)


def read_model(model, scene):
    semantic_paths = glob.glob(
        os.path.join(model, "evaluation", "semantic", "*", "metrics_lerf.json")
    )
    rgb_path = os.path.join(model, "evaluation", "rgb.json")
    manifests = glob.glob(os.path.join(model, "deploy", "*.manifest.json"))
    if not semantic_paths or not os.path.isfile(rgb_path) or not manifests:
        return None
    semantic = load(semantic_paths[0])
    rgb = load(rgb_path)
    manifest = load(manifests[0])
    return {
        "rendered_miou": semantic["rendered_miou"],
        "localization_accuracy": semantic["localization_accuracy"],
        "psnr": rgb["metrics"]["psnr"],
        "deployment_bytes": manifest["one_scene_total_bytes"],
        "point_count": manifest["point_count"],
        "source_files": [semantic_paths[0], rgb_path, manifests[0]],
    }


def aggregate(rows):
    numeric = (
        "rendered_miou",
        "localization_accuracy",
        "psnr",
        "deployment_bytes",
        "point_count",
    )
    result = {key: mean(row[key] for row in rows.values()) for key in numeric}
    result["deployment_mb_decimal"] = result["deployment_bytes"] / 1e6
    result["scene_count"] = len(rows)
    result["missing_scenes"] = [scene for scene in SCENES if scene not in rows]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--new-root", default="/data2/jian/outputs/wacv27_section6_20260819"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    experiment = "/data2/jian/outputs/wacv27_experiments"
    specifications = {
        "host": (
            experiment + "/clunogs_mvp/lerf_ovs/{scene}/uncompressed"
        ),
        "sparsification_only": args.new_root + "/spars_only/{scene}",
        "vq_only": args.new_root + "/vq_only/{scene}",
        "full": (
            experiment + "/clunogs_mvp/lerf_ovs/{scene}/joint_semantic"
        ),
        "without_semantic_guidance": (
            experiment
            + "/core_ablations/lerf_ovs/{scene}/joint_stop_semantic_support_grad"
        ),
        "without_dual_feedback": args.new_root + "/no_dual/{scene}",
        "sparsification_then_vq": (
            experiment + "/clunogs_sequential/lerf_ovs/{scene}/semantic_vq"
        ),
        "vq_then_sparsification": (
            args.new_root + "/vq_then_spars/{scene}/spars_second"
        ),
    }

    per_variant = {}
    macro = {}
    for variant, template in specifications.items():
        rows = {}
        for scene in SCENES:
            row = read_model(template.format(scene=scene), scene)
            if row is not None:
                rows[scene] = row
        per_variant[variant] = rows
        if rows:
            macro[variant] = aggregate(rows)

    result = {
        "protocol": (
            "four-scene macro; common fresh-reload LERF evaluator; storage in "
            "decimal MB; partial variants explicitly list missing scenes"
        ),
        "per_variant": per_variant,
        "macro": macro,
    }
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(macro, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
