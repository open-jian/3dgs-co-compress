"""Build table-ready FCGS + semantic-sidecar metrics for four LERF scenes."""

import argparse
import glob
import json
import os
import re
from datetime import datetime
from statistics import mean


DEFAULT_SCENES = ("ramen", "figurines", "teatime", "waldo_kitchen")
RGB_RE = re.compile(
    r"Evaluation results:\s*psnr:\s*([0-9.eE+-]+),\s*"
    r"ssim:\s*([0-9.eE+-]+),\s*lpips:\s*([0-9.eE+-]+),\s*Ll1:\s*([0-9.eE+-]+)"
)
ELAPSED_RE = re.compile(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([^\s]+)")
INTERNAL_ENCODE_RE = re.compile(r"^time:\s*([0-9.eE+-]+)\s*$", re.MULTILINE)


def _read(path):
    with open(path, "r", errors="replace") as source:
        return source.read()


def _load_json(path):
    with open(path, "r") as source:
        return json.load(source)


def _elapsed_seconds(value):
    fields = [float(part) for part in value.split(":")]
    if len(fields) == 2:
        return 60.0 * fields[0] + fields[1]
    if len(fields) == 3:
        return 3600.0 * fields[0] + 60.0 * fields[1] + fields[2]
    raise ValueError("unsupported elapsed time: {}".format(value))


def _wall_seconds(log_path):
    matches = ELAPSED_RE.findall(_read(log_path))
    if not matches:
        raise ValueError("missing wall-clock timing in {}".format(log_path))
    return _elapsed_seconds(matches[-1])


def _successful_validation_log(scene_dir):
    candidates = sorted(glob.glob(os.path.join(scene_dir, "logs", "validate*.log")))
    successful = []
    for path in candidates:
        text = _read(path)
        if "Evaluation results:" in text and "Exit status: 0" in text:
            successful.append(path)
    if not successful:
        raise ValueError("no successful decoded RGB validation in {}".format(scene_dir))
    return max(successful, key=os.path.getmtime)


def _find_newest(root, scene, filename, excluded_components=()):
    candidates = glob.glob(os.path.join(root, "**", scene, "**", filename), recursive=True)
    candidates += glob.glob(
        os.path.join(root, "**", "{}_*".format(scene), "**", filename),
        recursive=True,
    )
    if not candidates:
        candidates = glob.glob(os.path.join(root, scene, "**", filename), recursive=True)
    if not candidates:
        # ``root`` may already be the exact scene/operating-point directory.
        candidates = glob.glob(os.path.join(root, "**", filename), recursive=True)
    candidates = list(set(candidates))
    if excluded_components:
        blocked = set(excluded_components)
        candidates = [
            path for path in candidates
            if not blocked.intersection(
                os.path.relpath(path, root).split(os.sep)
            )
        ]
    return max(candidates, key=os.path.getmtime) if candidates else None


def _find_scene_log(root, scene):
    if not root:
        return None
    candidates = glob.glob(os.path.join(root, "**", "{}*.log".format(scene)), recursive=True)
    candidates += glob.glob(os.path.join(root, "**", scene, "**", "*.log"), recursive=True)
    candidates = list(set(candidates))
    return max(candidates, key=os.path.getmtime) if candidates else None


def _rgb_build_seconds(log_path):
    text = _read(log_path)
    starts = re.findall(r"\[([^\]]+)\]\s+train\s+(?:source|output)=", text)
    # The scene is deployable when training/saving finishes and rendering
    # starts; do not charge the baseline for our offline render/evaluation.
    ends = re.findall(r"\[([^\]]+)\]\s+render\s+output=", text)
    if not ends:
        ends = re.findall(r"\[([^\]]+)\]\s+complete\s+output=", text)
    if not starts or not ends:
        raise ValueError("missing RGB build start/complete markers in {}".format(log_path))
    start = datetime.strptime(starts[0], "%Y-%m-%dT%H:%M:%S%z")
    end = datetime.strptime(ends[-1], "%Y-%m-%dT%H:%M:%S%z")
    return float((end - start).total_seconds())


def _scene_record(codec_root, semantic_root, eval_root, scene,
                  rgb_build_log_root=None, semantic_log_root=None,
                  operating_point=None):
    scene_dir = os.path.join(codec_root, "lerf_ovs", scene)
    if operating_point:
        scene_dir = os.path.join(scene_dir, operating_point)
    semantic_scene_candidates = (
        os.path.join(semantic_root, "lerf_ovs", scene),
        os.path.join(semantic_root, scene),
    )
    semantic_scene_root = next(
        (path for path in semantic_scene_candidates if os.path.isdir(path)),
        semantic_root,
    )
    if operating_point:
        semantic_scene_root = os.path.join(semantic_scene_root, operating_point)
    excluded_operating_points = () if operating_point else ("light", "aggressive")
    storage = _load_json(os.path.join(scene_dir, "storage.json"))
    validate_path = _successful_validation_log(scene_dir)
    validate_text = _read(validate_path)
    match = RGB_RE.search(validate_text)
    if match is None:
        raise ValueError("missing decoded RGB metrics in {}".format(validate_path))
    psnr, ssim, lpips, l1 = map(float, match.groups())

    encode_log = os.path.join(scene_dir, "logs", "encode.log")
    decode_log = os.path.join(scene_dir, "logs", "decode.log")
    internal_match = INTERNAL_ENCODE_RE.search(_read(encode_log))
    record = {
        "scene": scene,
        "codec_roundtrip_verified": all(
            os.path.exists(os.path.join(scene_dir, marker))
            for marker in (".encode.complete", ".decode.complete", ".codec.complete")
        ),
        "decoded_rgb": {
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "l1": l1,
            "validation_log": validate_path,
        },
        "codec_timing_seconds": {
            "encode_wall": _wall_seconds(encode_log),
            "decode_wall": _wall_seconds(decode_log),
            "encode_internal": float(internal_match.group(1)) if internal_match else None,
        },
        "storage": {
            "scene_bitstream_bytes": int(storage["scene_bitstream_bytes"]),
            "shared_fcgs_checkpoint_bytes": int(storage["shared_checkpoint_bytes"]),
        },
    }
    rgb_build_log = _find_scene_log(rgb_build_log_root, scene)
    if rgb_build_log:
        record["build_timing_seconds"] = {
            "rgb_reconstruction": _rgb_build_seconds(rgb_build_log),
            "rgb_reconstruction_log": rgb_build_log,
        }
    semantic_log = _find_scene_log(semantic_log_root, scene)
    if semantic_log:
        try:
            semantic_seconds = _wall_seconds(semantic_log)
        except ValueError:
            semantic_seconds = None
        if semantic_seconds is not None:
            record.setdefault("build_timing_seconds", {})["semantic_construction"] = semantic_seconds
            record["build_timing_seconds"]["semantic_construction_log"] = semantic_log
    if (
        "semantic_construction" not in record.get("build_timing_seconds", {})
    ):
        cfg_path = _find_newest(
            semantic_scene_root, scene, "cfg_args", excluded_operating_points
        )
        checkpoint_path = _find_newest(
            semantic_scene_root, scene, "chkpnt*.pth", excluded_operating_points
        )
        if cfg_path and checkpoint_path:
            # run_fcgs_langsplatv2_scene.sh writes cfg_args immediately before
            # initialization/training and closes the checkpoint at completion.
            # Preserve this fallback provenance explicitly when the external
            # /usr/bin/time log was not part of the rsync payload.
            semantic_seconds = os.path.getmtime(checkpoint_path) - os.path.getmtime(cfg_path)
            if semantic_seconds >= 0:
                timing = record.setdefault("build_timing_seconds", {})
                timing["semantic_construction"] = float(semantic_seconds)
                timing["semantic_construction_source"] = "cfg_args_to_checkpoint_mtime"
                timing["semantic_construction_cfg"] = cfg_path
                timing["semantic_construction_checkpoint"] = checkpoint_path

    manifest_path = _find_newest(
        semantic_scene_root, scene, "*.manifest.json", excluded_operating_points
    )
    if manifest_path:
        manifest = _load_json(manifest_path)
        if manifest.get("format") == "fcgs.semantic-sidecar":
            sidecar_bytes = int(manifest["semantic_sidecar_bytes"])
            shared_semantic_bytes = int(manifest.get("shared_semantic_bytes", 0))
            record["semantic_sidecar"] = {
                "bytes": sidecar_bytes,
                "shared_semantic_bytes": shared_semantic_bytes,
                "contains_geometry": bool(manifest.get("contains_geometry", True)),
                "point_count": int(manifest["point_count"]),
                "topk": int(manifest["topk"]),
                "manifest": manifest_path,
            }
            record["storage"].update({
                "semantic_sidecar_bytes": sidecar_bytes,
                "shared_semantic_bytes": shared_semantic_bytes,
                "scene_payload_bytes": int(storage["scene_bitstream_bytes"]) + sidecar_bytes,
                "shared_payload_bytes": (
                    int(storage["shared_checkpoint_bytes"]) + shared_semantic_bytes
                ),
                "deployment_total_bytes": (
                    int(storage["scene_bitstream_bytes"])
                    + int(storage["shared_checkpoint_bytes"])
                    + sidecar_bytes
                    + shared_semantic_bytes
                ),
            })
            record["gaussian_count"] = int(manifest["point_count"])
            reload_path = _find_newest(
                semantic_scene_root, scene, "semantic_sidecar_reload.json",
                excluded_operating_points,
            )
            if reload_path:
                reload_result = _load_json(reload_path)
                record["semantic_sidecar"]["reload"] = {
                    "reload_ok": bool(reload_result["reload_ok"]),
                    "geometry_match": bool(reload_result["geometry_match"]),
                    "indices_exact": bool(reload_result.get("indices_exact", False)),
                    "semantic_feature_cosine_mean": reload_result.get(
                        "semantic_feature_cosine_mean"
                    ),
                    "semantic_feature_cosine_min": reload_result.get(
                        "semantic_feature_cosine_min"
                    ),
                    "validation_json": reload_path,
                }

    if "build_timing_seconds" in record:
        timing = record["build_timing_seconds"]
        if "rgb_reconstruction" in timing and "semantic_construction" in timing:
            timing["end_to_end"] = (
                timing["rgb_reconstruction"]
                + record["codec_timing_seconds"]["encode_wall"]
                + record["codec_timing_seconds"]["decode_wall"]
                + timing["semantic_construction"]
            )

    metrics_path = _find_newest(
        eval_root, scene, "metrics_lerf.json", excluded_operating_points
    )
    if metrics_path:
        metrics = _load_json(metrics_path)
        record["semantics"] = {
            "rendered_miou": float(metrics["rendered_miou"]),
            "localization_correct": int(metrics["localization_correct"]),
            "localization_total": int(metrics["localization_total"]),
            "localization_accuracy": float(metrics["localization_accuracy"]),
            "metrics_json": metrics_path,
        }
        if "timing" in metrics:
            semantic_timing = dict(metrics["timing"])
            semantic_timing["end_to_end_query_ms_per_view"] = (
                float(semantic_timing["semantic_render_ms_per_view"])
                + float(semantic_timing["query_batch_ms_per_view"])
            )
            total_query_ms = (
                semantic_timing["end_to_end_query_ms_per_view"]
                * int(semantic_timing["annotated_views"])
            )
            semantic_timing["end_to_end_query_ms_per_prompt"] = (
                total_query_ms / int(metrics["localization_total"])
            )
            record["semantic_timing"] = semantic_timing
    return record


def _macro(records):
    macro = {
        "scene_count": len(records),
        "decoded_rgb": {
            key: mean(record["decoded_rgb"][key] for record in records)
            for key in ("psnr", "ssim", "lpips", "l1")
        },
        "codec_timing_seconds": {
            key: mean(record["codec_timing_seconds"][key] for record in records)
            for key in ("encode_wall", "decode_wall", "encode_internal")
        },
        "storage": {
            key: mean(record["storage"][key] for record in records)
            for key in ("scene_bitstream_bytes", "shared_fcgs_checkpoint_bytes")
        },
    }
    if all("deployment_total_bytes" in record["storage"] for record in records):
        for key in (
            "semantic_sidecar_bytes", "shared_semantic_bytes", "scene_payload_bytes",
            "shared_payload_bytes", "deployment_total_bytes"
        ):
            macro["storage"][key] = mean(record["storage"][key] for record in records)
        macro["storage_mib"] = {
            key.replace("_bytes", "_mib"): value / (1024.0 * 1024.0)
            for key, value in macro["storage"].items()
        }
        macro["gaussian_count"] = mean(record["gaussian_count"] for record in records)
    if all(
        "build_timing_seconds" in record
        and "rgb_reconstruction" in record["build_timing_seconds"]
        for record in records
    ):
        macro["build_timing_seconds"] = {
            "rgb_reconstruction": mean(
                record["build_timing_seconds"]["rgb_reconstruction"]
                for record in records
            )
        }
    if all(
        "build_timing_seconds" in record
        and "end_to_end" in record["build_timing_seconds"]
        for record in records
    ):
        macro.setdefault("build_timing_seconds", {}).update({
            key: mean(record["build_timing_seconds"][key] for record in records)
            for key in ("semantic_construction", "end_to_end")
        })
    if all("semantics" in record for record in records):
        macro["semantics"] = {
            "rendered_miou_scene_macro": mean(
                record["semantics"]["rendered_miou"] for record in records
            ),
            "localization_accuracy_scene_macro": mean(
                record["semantics"]["localization_accuracy"] for record in records
            ),
            "localization_correct_pooled": sum(
                record["semantics"]["localization_correct"] for record in records
            ),
            "localization_total_pooled": sum(
                record["semantics"]["localization_total"] for record in records
            ),
        }
        macro["semantics"]["localization_accuracy_pooled"] = (
            macro["semantics"]["localization_correct_pooled"]
            / macro["semantics"]["localization_total_pooled"]
        )
    if all("semantic_timing" in record for record in records):
        timing_keys = (
            "representation_load_seconds",
            "semantic_render_ms_per_view",
            "query_batch_ms_per_view",
            "query_ms_per_prompt",
            "end_to_end_query_ms_per_view",
            "end_to_end_query_ms_per_prompt",
        )
        macro["semantic_timing"] = {
            key: mean(record["semantic_timing"][key] for record in records)
            for key in timing_keys
        }
    return macro


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec-root", required=True)
    parser.add_argument("--semantic-root", required=True)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--rgb-build-log-root")
    parser.add_argument("--semantic-log-root")
    parser.add_argument(
        "--operating-point",
        choices=("light", "aggressive"),
        help="optional FCGS operating-point subdirectory; omit for medium",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    args = parser.parse_args()

    records = [
        _scene_record(
            args.codec_root, args.semantic_root, args.eval_root, scene,
            rgb_build_log_root=args.rgb_build_log_root,
            semantic_log_root=args.semantic_log_root,
            operating_point=args.operating_point,
        )
        for scene in args.scenes
    ]
    macro = _macro(records)
    table_row = {
        "method": "FCGS->LangSplatV2",
        "rendered_miou": macro.get("semantics", {}).get("rendered_miou_scene_macro"),
        "rendered_miou_percent": (
            100.0 * macro["semantics"]["rendered_miou_scene_macro"]
            if "semantics" in macro else None
        ),
        "localization_accuracy": macro.get("semantics", {}).get(
            "localization_accuracy_scene_macro"
        ),
        "localization_accuracy_percent": (
            100.0 * macro["semantics"]["localization_accuracy_scene_macro"]
            if "semantics" in macro else None
        ),
        "psnr": macro["decoded_rgb"]["psnr"],
        "lpips": macro["decoded_rgb"]["lpips"],
        "gaussian_count": macro.get("gaussian_count"),
        "encode_wall_seconds": macro["codec_timing_seconds"]["encode_wall"],
        "decode_wall_seconds": macro["codec_timing_seconds"]["decode_wall"],
        "query_latency_ms_per_view": macro.get("semantic_timing", {}).get(
            "end_to_end_query_ms_per_view"
        ),
        "query_latency_ms_per_prompt": macro.get("semantic_timing", {}).get(
            "end_to_end_query_ms_per_prompt"
        ),
    }
    if "storage_mib" in macro:
        table_row.update({
            "scene_mib": macro["storage_mib"]["scene_payload_mib"],
            "shared_mib": macro["storage_mib"]["shared_payload_mib"],
            "total_at_1_mib": macro["storage_mib"]["deployment_total_mib"],
            "storage_s_sh_t_at_1_mib": "{:.2f} / {:.2f} / {:.2f}".format(
                macro["storage_mib"]["scene_payload_mib"],
                macro["storage_mib"]["shared_payload_mib"],
                macro["storage_mib"]["deployment_total_mib"],
            ),
        })
    if "build_timing_seconds" in macro and "end_to_end" in macro["build_timing_seconds"]:
        table_row["end_to_end_seconds"] = macro["build_timing_seconds"]["end_to_end"]
        table_row["end_to_end_minutes"] = macro["build_timing_seconds"]["end_to_end"] / 60.0

    summary = {
        "method": "FCGS + LangSplatV2 semantic sidecar",
        "aggregation": "Unweighted four-scene macro; localization pooled also reported",
        "size_accounting": (
            "scene FCGS bitstream + shared FCGS checkpoint + scene semantic sidecar "
            "+ shared semantic components"
        ),
        "timing_definition": "wall clock including process and file I/O",
        "per_scene": records,
        "macro": macro,
        "table_row": table_row,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
