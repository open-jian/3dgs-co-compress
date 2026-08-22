#!/usr/bin/env python3
"""Build the Section 5.3 four-scene summary from fresh-reload artifacts."""

import argparse
import glob
import json
import os
from statistics import mean


SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")


def load(path):
    with open(path, "r") as handle:
        return json.load(handle)


def modular_row(root, method, scene):
    base = os.path.join(root, "modular_semantics", method, scene)
    semantic_path = glob.glob(
        os.path.join(base, "evaluation", "semantic", "*", "metrics_lerf.json")
    )[0]
    rgb_path = os.path.join(base, "evaluation", "rgb.json")
    semantic = load(semantic_path)
    rgb = load(rgb_path)
    host_iteration = glob.glob(
        os.path.join(root, "modular_hosts", method, scene, "point_cloud", "iteration_*")
    )[0]
    payload = [os.path.join(host_iteration, "point_cloud.ply")]
    if method == "compgs":
        payload.extend(
            os.path.join(host_iteration, name)
            for name in ("kmeans_args.npy", "kmeans_centers.pth", "kmeans_inds.bin")
        )
    payload.append(os.path.join(base, "deploy", scene + ".semantic.sidecar.pth"))
    return {
        "rendered_miou": semantic["rendered_miou"],
        "localization_accuracy": semantic["localization_accuracy"],
        "psnr": rgb["metrics"]["psnr"],
        "point_count": rgb["point_count"],
        "deployment_bytes": sum(os.path.getsize(path) for path in payload),
        "deployment_files": payload,
        "metric_files": [semantic_path, rgb_path],
    }


def compact_row(experiment_root, scene, variant, artifact_suffix):
    base = os.path.join(experiment_root, scene, variant)
    semantic_path = glob.glob(
        os.path.join(base, "evaluation", "semantic", "*", "metrics_lerf.json")
    )[0]
    rgb_path = os.path.join(base, "evaluation", "rgb.json")
    manifest_path = os.path.join(base, "deploy", scene + artifact_suffix)
    semantic = load(semantic_path)
    rgb = load(rgb_path)
    manifest = load(manifest_path)
    return {
        "rendered_miou": semantic["rendered_miou"],
        "localization_accuracy": semantic["localization_accuracy"],
        "psnr": rgb["metrics"]["psnr"],
        "point_count": manifest["point_count"],
        "deployment_bytes": manifest["one_scene_total_bytes"],
        "deployment_files": [manifest_path],
        "metric_files": [semantic_path, rgb_path],
    }


def aggregate(rows):
    keys = (
        "rendered_miou",
        "localization_accuracy",
        "psnr",
        "point_count",
        "deployment_bytes",
    )
    return {key: mean(row[key] for row in rows.values()) for key in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table-root",
        default="/data2/jian/outputs/wacv27_table_runs_20260817",
    )
    parser.add_argument(
        "--experiment-root",
        default="/data2/jian/outputs/wacv27_experiments/clunogs_mvp/lerf_ovs",
    )
    parser.add_argument(
        "--fcgs-summary",
        default=(
            "/data2/jian/outputs/wacv27_experiments/fcgs_langsplatv2/"
            "lerf_ovs/fcgs_lerf_table_summary.json"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    per_method = {}
    per_method["LangSplatV2 host"] = {
        scene: compact_row(
            args.experiment_root,
            scene,
            "uncompressed",
            ".uncompressed.pth.manifest.json",
        )
        for scene in SCENES
    }
    per_method["ClunoGS"] = {
        scene: compact_row(
            args.experiment_root,
            scene,
            "joint_semantic",
            ".compact.pth.manifest.json",
        )
        for scene in SCENES
    }
    for method in ("gaussianspa", "compgs"):
        per_method[method] = {
            scene: modular_row(args.table_root, method, scene) for scene in SCENES
        }

    fcgs = load(args.fcgs_summary)
    fcgs_rows = {}
    for row in fcgs["per_scene"]:
        fcgs_rows[row["scene"]] = {
            "rendered_miou": row["semantics"]["rendered_miou"],
            "localization_accuracy": row["semantics"]["localization_accuracy"],
            "psnr": row["decoded_rgb"]["psnr"],
            "point_count": row["gaussian_count"],
            "deployment_bytes": row["storage"]["deployment_total_bytes"],
            "deployment_files": [args.fcgs_summary],
            "metric_files": [
                row["semantics"]["metrics_json"],
                row["decoded_rgb"]["validation_log"],
            ],
        }
    per_method["fcgs"] = fcgs_rows

    macro = {method: aggregate(rows) for method, rows in per_method.items()}
    host_bytes = macro["LangSplatV2 host"]["deployment_bytes"]
    for row in macro.values():
        row["deployment_mb_decimal"] = row["deployment_bytes"] / 1e6
        row["compression_ratio_vs_host"] = host_bytes / row["deployment_bytes"]

    result = {
        "protocol": (
            "four-scene macro; fresh-reload metrics; complete deployment payload; "
            "storage uses decimal MB"
        ),
        "scenes": list(SCENES),
        "per_method": per_method,
        "macro": macro,
    }
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(macro, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
