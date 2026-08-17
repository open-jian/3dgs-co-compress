#!/usr/bin/env python3
"""Aggregate fresh-reload LERF compact evaluations into one auditable JSON."""

import argparse
from datetime import datetime
import json
import re
from pathlib import Path
from statistics import mean


SCENES = ("ramen", "figurines", "teatime", "waldo_kitchen")
ELAPSED_RE = re.compile(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)")
STAMP_RE = re.compile(r"^\[([^]]+)\] (train|render) ", re.MULTILINE)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def elapsed_seconds(value: str) -> float:
    fields = value.split(":")
    if len(fields) == 2:
        minutes, seconds = fields
        return 60.0 * float(minutes) + float(seconds)
    if len(fields) == 3:
        hours, minutes, seconds = fields
        return 3600.0 * float(hours) + 60.0 * float(minutes) + float(seconds)
    raise ValueError(f"Unsupported elapsed time: {value}")


def elapsed_values(path: Path):
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return [elapsed_seconds(value) for value in ELAPSED_RE.findall(text)]


def rgb_build_seconds(path: Path) -> float:
    """Measure reconstruction only, stopping before render/evaluation."""
    if not path.is_file():
        return 0.0
    text = path.read_text(encoding="utf-8", errors="replace")
    stamps = {kind: datetime.fromisoformat(value) for value, kind in STAMP_RE.findall(text)}
    if "train" in stamps and "render" in stamps:
        return (stamps["render"] - stamps["train"]).total_seconds()
    return sum(elapsed_values(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--training-log-root", type=Path)
    parser.add_argument(
        "--phase-log",
        action="append",
        default=[],
        metavar="SCENE=PATH[:first|all]",
        help=(
            "override method-phase logs for one scene; repeat to combine stages "
            "or recover a resumed run"
        ),
    )
    parser.add_argument(
        "--rgb-log-root",
        type=Path,
        default=Path("/data2/jian/outputs/rgb_3dgs/logs/train_lerf_ovs"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    phase_overrides = {}
    for specification in args.phase_log:
        if "=" not in specification:
            parser.error("--phase-log must be SCENE=PATH[:first|all]")
        scene, path_and_selector = specification.split("=", 1)
        if scene not in SCENES:
            parser.error("unknown --phase-log scene: " + scene)
        selector = "all"
        path_text = path_and_selector
        if path_and_selector.endswith(":first"):
            path_text = path_and_selector[:-6]
            selector = "first"
        elif path_and_selector.endswith(":all"):
            path_text = path_and_selector[:-4]
        phase_overrides.setdefault(scene, []).append((Path(path_text), selector))

    per_scene = {}
    missing = []
    for scene in SCENES:
        model_root = args.experiment_root / scene / args.variant
        semantic_path = (
            model_root / "evaluation" / "semantic" / f"{scene}_0" / "metrics_lerf.json"
        )
        rgb_path = model_root / "evaluation" / "rgb.json"
        manifest_path = model_root / "deploy" / f"{scene}.compact.pth.manifest.json"
        required = (semantic_path, rgb_path, manifest_path)
        if not all(path.is_file() for path in required):
            missing.append(scene)
            continue

        semantic = load_json(semantic_path)
        rgb = load_json(rgb_path)
        manifest = load_json(manifest_path)
        rgb_seconds = rgb_build_seconds(args.rgb_log_root / f"{scene}.log")
        method_seconds = []
        method_phase_sources = []
        if scene in phase_overrides:
            for phase_path, selector in phase_overrides[scene]:
                values = elapsed_values(phase_path)
                if selector == "first":
                    values = values[:1]
                method_seconds.extend(values)
                method_phase_sources.append({
                    "path": str(phase_path),
                    "selector": selector,
                    "seconds": values,
                })
        elif args.training_log_root:
            phase_path = args.training_log_root / f"{scene}.log"
            method_seconds = elapsed_values(phase_path)
            method_phase_sources.append({
                "path": str(phase_path),
                "selector": "all",
                "seconds": method_seconds,
            })

        per_scene[scene] = {
            "rendered_miou": semantic["rendered_miou"],
            "localization_correct": semantic["localization_correct"],
            "localization_total": semantic["localization_total"],
            "localization_accuracy": semantic["localization_accuracy"],
            "psnr": rgb["metrics"]["psnr"],
            "ssim": rgb["metrics"]["ssim"],
            "lpips": rgb["metrics"]["lpips"],
            "scene_bytes": manifest["scene_bytes"],
            "shared_bytes": manifest["shared_bytes"],
            "total_at_one_bytes": manifest["one_scene_total_bytes"],
            "point_count": manifest["point_count"],
            "query_ms_per_prompt": semantic["timing"]["query_ms_per_prompt"],
            "rgb_build_seconds": rgb_seconds,
            "method_phase_seconds": method_seconds,
            "method_phase_sources": method_phase_sources,
            "common_preprocessed_scene_build_seconds": rgb_seconds + sum(method_seconds),
            "source_files": [str(path) for path in required],
        }

    if missing and not args.allow_partial:
        raise SystemExit("Missing finalized scenes: " + ", ".join(missing))
    if not per_scene:
        raise SystemExit("No finalized scenes found")

    rows = list(per_scene.values())
    correct = sum(row["localization_correct"] for row in rows)
    total = sum(row["localization_total"] for row in rows)
    macro = {
        "scene_count": len(rows),
        "rendered_miou": mean(row["rendered_miou"] for row in rows),
        "localization_scene_macro": mean(row["localization_accuracy"] for row in rows),
        "localization_pooled": correct / total,
        "localization_correct": correct,
        "localization_total": total,
        "psnr": mean(row["psnr"] for row in rows),
        "ssim": mean(row["ssim"] for row in rows),
        "lpips": mean(row["lpips"] for row in rows),
        "scene_mib": mean(row["scene_bytes"] for row in rows) / 2**20,
        "shared_mib": mean(row["shared_bytes"] for row in rows) / 2**20,
        "total_at_one_mib": mean(row["total_at_one_bytes"] for row in rows) / 2**20,
        "point_count": mean(row["point_count"] for row in rows),
        "query_ms_per_prompt": mean(row["query_ms_per_prompt"] for row in rows),
        "common_preprocessed_scene_build_minutes": mean(
            row["common_preprocessed_scene_build_seconds"] for row in rows
        ) / 60.0,
    }
    result = {
        "protocol": "LERF four-scene common evaluator; scene-macro unless marked pooled",
        "build_scope": "RGB reconstruction plus logged method phases; common 2D supervision extraction excluded",
        "experiment_root": str(args.experiment_root),
        "variant": args.variant,
        "missing_scenes": missing,
        "per_scene": per_scene,
        "macro": macro,
    }
    output = args.output or args.experiment_root / f"{args.variant}_lerf_compact_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(macro, indent=2, sort_keys=True))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
