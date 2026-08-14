#!/usr/bin/env python3
"""Inference-only query-speed benchmark for the released LangSplatV2 path.

This is a third-party benchmark harness, not an official benchmark script from
the LangSplatV2 authors.  It loads a scene and the three semantic checkpoints
once, combines them exactly as the released quick evaluator intends, encodes
the text query before timing, and then measures GPU work with CUDA events.

The timed stages deliberately exclude scene construction, checkpoint and image
I/O, top-k coefficient extraction, and OpenCLIP text encoding.  The
``relevancy_postprocess`` stage follows the released LERF *segmentation*
post-processing operations but omits ground-truth IoU computation, which is an
evaluation operation rather than query inference.
"""

from __future__ import print_function

import argparse
import datetime
import json
import os
import random
import subprocess
import tempfile
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from eval.openclip_encoder import OpenCLIPNetwork
from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.vq_utils import get_weights_and_indices


SEMANTIC_LEVELS = 3
CODEBOOK_SIZE = 64
FEATURE_DIMENSION = 512
TOPK = 4
SPARSE_RENDER_CHANNELS = SEMANTIC_LEVELS * TOPK
COEFFICIENT_CHANNELS = SEMANTIC_LEVELS * CODEBOOK_SIZE


def _timestamp_now():
    return datetime.datetime.now().astimezone().isoformat()


def _git_metadata(repo_path):
    metadata = {"commit": None, "dirty": None}
    try:
        metadata["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_path)
        ).decode("utf-8").strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(repo_path)
        ).decode("utf-8")
        metadata["dirty"] = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return metadata


def _driver_version():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ]
        ).decode("utf-8")
        versions = sorted(set(line.strip() for line in output.splitlines() if line.strip()))
        return versions[0] if len(versions) == 1 else versions
    except (OSError, subprocess.CalledProcessError):
        return None


def _checkpoint_metadata(path, first_iteration, model_parameters):
    stat_result = path.stat()
    modified = datetime.datetime.fromtimestamp(
        stat_result.st_mtime, tz=datetime.timezone.utc
    ).isoformat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat_result.st_size),
        "mtime_utc": modified,
        "sha256": None,
        "identity_mode": "absolute_path_and_file_metadata",
        "checkpoint_iteration_stored": int(first_iteration),
        "model_parameter_tuple_length": len(model_parameters),
    }


def _decode_released_quick(language_feature_weight_map, codebooks):
    """The einsum and normalization from render_language_feature_map_quick."""

    dimension, height, width = language_feature_weight_map.shape
    if dimension != COEFFICIENT_CHANNELS:
        raise RuntimeError(
            "Expected {} coefficient channels, got {}".format(
                COEFFICIENT_CHANNELS, dimension
            )
        )
    coefficient_map = language_feature_weight_map.view(
        SEMANTIC_LEVELS, CODEBOOK_SIZE, height, width
    ).view(SEMANTIC_LEVELS, CODEBOOK_SIZE, height * width)
    language_codebooks = codebooks.permute(0, 2, 1)
    feature_map = torch.einsum(
        "ldk,lkn->ldn", language_codebooks, coefficient_map
    ).view(SEMANTIC_LEVELS, FEATURE_DIMENSION, height, width)
    return feature_map / (feature_map.norm(dim=1, keepdim=True) + 1e-10)


def _released_lerf_segmentation_postprocess(
    feature_map, clip_model, mask_threshold, average_pool, mask_pool
):
    """Released LERF segmentation query path without any ground-truth metric.

    Included operations are CLIP relevancy, the released 29x29 smoothing and
    0.5 residual blend, per-level min/max transformation, binary threshold,
    released 7x7 mask smoothing, and semantic-level selection by peak score.
    Text embeddings are already resident in ``clip_model`` before this call.
    """

    semantic_map = feature_map.permute(0, 2, 3, 1)
    valid_map = clip_model.get_max_across_quick(semantic_map)
    num_levels, num_prompts, _, _ = valid_map.shape
    if num_levels != SEMANTIC_LEVELS or num_prompts != 1:
        raise RuntimeError(
            "Benchmark expects {} levels and one positive query, got {} and {}".format(
                SEMANTIC_LEVELS, num_levels, num_prompts
            )
        )

    masks = []
    level_scores = []
    for level_index in range(num_levels):
        relevancy = valid_map[level_index, 0]
        averaged = average_pool(relevancy.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        blended = 0.5 * (averaged + relevancy)

        normalized = blended - torch.min(blended)
        normalized = normalized / (torch.max(normalized) + 1e-9)
        normalized = normalized * 2.0 - 1.0
        normalized = torch.clip(normalized, 0.0, 1.0)

        mask = (normalized > mask_threshold).to(torch.uint8)
        smoothed = mask_pool(mask.float().unsqueeze(0).unsqueeze(0))
        masks.append((smoothed > 0.5).to(torch.uint8).squeeze(0).squeeze(0))
        level_scores.append(torch.max(blended))

    masks_tensor = torch.stack(masks, dim=0)
    scores_tensor = torch.stack(level_scores, dim=0)
    selected_level = torch.argmax(scores_tensor)
    selected_mask = masks_tensor[selected_level]
    return selected_mask, selected_level, scores_tensor[selected_level]


def _summarize_times(samples_ms):
    values = np.asarray(samples_ms, dtype=np.float64)
    mean_ms = float(np.mean(values))
    median_ms = float(np.median(values))
    p95_ms = float(np.percentile(values, 95))
    return {
        "sample_count": int(values.size),
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "fps_from_mean_ms": float(1000.0 / mean_ms),
        "fps_from_median_ms": float(1000.0 / median_ms),
        "fps_from_p95_ms": float(1000.0 / p95_ms),
    }


def _benchmark_cuda_callable(function, warmup, repeat, device):
    last_output = None
    with torch.no_grad():
        for _ in range(warmup):
            last_output = function()
        torch.cuda.synchronize(device)
        del last_output

        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        samples_ms = []
        for _ in range(repeat):
            start_event.record()
            last_output = function()
            end_event.record()
            end_event.synchronize()
            samples_ms.append(float(start_event.elapsed_time(end_event)))
        del last_output
        torch.cuda.synchronize(device)

    summary = _summarize_times(samples_ms)
    summary.update(
        {
            "warmup_count": int(warmup),
            "baseline_memory_allocated_bytes": int(baseline_allocated),
            "baseline_memory_reserved_bytes": int(baseline_reserved),
            "peak_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    )
    return summary


def _validate_arguments(args):
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    if args.camera_index < 0:
        raise ValueError("--camera-index must be non-negative")
    if not (0.0 <= args.mask_threshold <= 1.0):
        raise ValueError("--mask-threshold must be in [0, 1]")
    if not args.positive.strip():
        raise ValueError("--positive must not be empty")


def _build_parser(repo_path):
    workspace_path = repo_path.parent.parent
    parser = argparse.ArgumentParser(
        description="Third-party, inference-only LangSplatV2 CUDA-event benchmark"
    )
    parser.add_argument("--scene-name", default="figurines")
    parser.add_argument("--model-index", default="0")
    parser.add_argument("--checkpoint-iteration", type=int, default=10000)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=workspace_path / "Data" / "langsplatv2_official_weights" / "output",
    )
    parser.add_argument(
        "--source-path",
        type=Path,
        default=None,
        help="Defaults to <workspace>/data/lerf_ovs/<scene-name>",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=194,
        help="Zero-based training-camera index; 194 is annotated frame_00195 for figurines",
    )
    parser.add_argument("--positive", default="waldo")
    parser.add_argument("--mask-threshold", type=float, default=0.4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=repo_path / "benchmark_query_speed.json",
    )
    return parser


def main():
    repo_path = Path(__file__).resolve().parent
    args = _build_parser(repo_path).parse_args()
    _validate_arguments(args)

    if args.source_path is None:
        args.source_path = (
            repo_path.parent.parent / "Data" / "lerf_ovs" / args.scene_name
        )
    args.source_path = args.source_path.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    args.output_json = args.output_json.resolve()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.device_index >= torch.cuda.device_count():
        raise ValueError(
            "--device-index {} is outside {} visible CUDA devices".format(
                args.device_index, torch.cuda.device_count()
            )
        )

    device = torch.device("cuda", args.device_index)
    torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.source_path.is_dir():
        raise FileNotFoundError(str(args.source_path))

    checkpoint_dirs = [
        args.checkpoint_root
        / "{}_{}_{}".format(args.scene_name, args.model_index, level)
        for level in range(1, SEMANTIC_LEVELS + 1)
    ]
    checkpoint_paths = [
        directory / "chkpnt{}.pth".format(args.checkpoint_iteration)
        for directory in checkpoint_dirs
    ]
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(str(checkpoint_path))

    started_at = _timestamp_now()
    setup_start = time.perf_counter()

    # These values match the released figurines checkpoint cfg_args.  The
    # temporary model directory prevents Scene from overwriting files in the
    # official checkpoint folders when it emits input.ply and cameras.json.
    dataset = Namespace(
        sh_degree=3,
        source_path=str(args.source_path),
        model_path=None,
        language_features_name="language_features",
        lf_path=str(args.source_path / "language_features"),
        images="images",
        resolution=-1,
        white_background=False,
        feature_level=1,
        data_device="cuda",
        eval=False,
    )
    pipeline = Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
    )
    render_options = Namespace(
        include_feature=True,
        quick_render=True,
        topk=TOPK,
    )

    combined_gaussians = GaussianModel(dataset.sh_degree)
    with tempfile.TemporaryDirectory(prefix="langsplatv2-speed-") as runtime_dir:
        dataset.model_path = runtime_dir
        scene = Scene(dataset, combined_gaussians, shuffle=False)
        views = scene.getTrainCameras()
        camera_count = len(views)
        if args.camera_index >= camera_count:
            raise ValueError(
                "--camera-index {} is outside {} cameras".format(
                    args.camera_index, camera_count
                )
            )
        view = views[args.camera_index]

    codebooks = []
    sparse_weights = []
    sparse_indices = []
    checkpoint_records = []
    point_counts = []
    source_logit_elements = 0

    for level_index, checkpoint_path in enumerate(checkpoint_paths):
        payload = torch.load(str(checkpoint_path), map_location=device)
        if not isinstance(payload, (tuple, list)) or len(payload) != 2:
            raise RuntimeError("Unexpected checkpoint payload: {}".format(checkpoint_path))
        model_parameters, first_iteration = payload
        if len(model_parameters) != 14:
            raise RuntimeError(
                "Expected a 14-field language checkpoint, got {} in {}".format(
                    len(model_parameters), checkpoint_path
                )
            )

        level_gaussians = combined_gaussians if level_index == 0 else GaussianModel(dataset.sh_degree)
        level_gaussians.restore(model_parameters, render_options, mode="test")
        point_count = int(level_gaussians.get_xyz.shape[0])
        point_counts.append(point_count)

        level_codebooks = level_gaussians._language_feature_codebooks
        if tuple(level_codebooks.shape) != (1, CODEBOOK_SIZE, FEATURE_DIMENSION):
            raise RuntimeError(
                "Checkpoint {} has incompatible codebook shape {}".format(
                    checkpoint_path, tuple(level_codebooks.shape)
                )
            )
        logits = level_gaussians._language_feature_logits
        if logits.shape[0] != point_count or logits.shape[1] != CODEBOOK_SIZE:
            raise RuntimeError(
                "Checkpoint {} has incompatible logits shape {}".format(
                    checkpoint_path, tuple(logits.shape)
                )
            )

        # utils.vq_utils contains the released mask/scatter extraction with a
        # device-placement compatibility fix, so this stays bit-identical to
        # the evaluator rather than using a reordered torch.topk shortcut.
        weights, indices = get_weights_and_indices(logits, TOPK)
        codebooks.append(level_codebooks.view(CODEBOOK_SIZE, FEATURE_DIMENSION))
        sparse_weights.append(weights)
        sparse_indices.append(indices + level_index * CODEBOOK_SIZE)
        source_logit_elements += int(logits.numel())
        checkpoint_records.append(
            _checkpoint_metadata(checkpoint_path, first_iteration, model_parameters)
        )

        del payload, model_parameters, logits, level_codebooks, weights, indices
        del level_gaussians

    if len(set(point_counts)) != 1:
        raise RuntimeError("The three checkpoints have different point counts: {}".format(point_counts))

    combined_gaussians._language_feature_codebooks = torch.stack(codebooks, dim=0)
    combined_gaussians._language_feature_weights = torch.cat(sparse_weights, dim=1)
    combined_indices = torch.cat(sparse_indices, dim=1)
    # Match the released evaluator's final dtype/layout conversion outside the
    # timed region.
    combined_gaussians._language_feature_indices = torch.from_numpy(
        combined_indices.detach().cpu().numpy()
    ).to(combined_gaussians._language_feature_weights.device)
    # Quick rendering consumes only the sparse weights/indices and merged
    # codebooks.  Release the dense level-1 logits retained by restore().
    combined_gaussians._language_feature_logits = None

    del codebooks, sparse_weights, sparse_indices, combined_indices
    del views, scene
    torch.cuda.empty_cache()

    background_color = [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(background_color, dtype=torch.float32, device=device)

    # Text-model construction and positive/negative text encoding happen here,
    # before any CUDA event is recorded.
    clip_model = OpenCLIPNetwork(device)
    clip_model.set_positives([args.positive])
    average_pool = torch.nn.AvgPool2d(
        kernel_size=29, stride=1, padding=14, count_include_pad=False
    ).to(device)
    mask_pool = torch.nn.AvgPool2d(
        kernel_size=7, stride=1, padding=3, count_include_pad=False
    ).to(device)
    torch.cuda.synchronize(device)

    setup_wall_seconds = float(time.perf_counter() - setup_start)

    def rasterizer_stage():
        output = render(
            view,
            combined_gaussians,
            pipeline,
            background,
            render_options,
        )
        return output["language_feature_weight_map"]

    def decode_stage(coefficient_map):
        return _decode_released_quick(
            coefficient_map, combined_gaussians._language_feature_codebooks
        )

    def postprocess_stage(feature_map):
        return _released_lerf_segmentation_postprocess(
            feature_map,
            clip_model,
            args.mask_threshold,
            average_pool,
            mask_pool,
        )

    benchmark_start = time.perf_counter()
    timings = {}
    timings["rasterizer"] = _benchmark_cuda_callable(
        rasterizer_stage, args.warmup, args.repeat, device
    )

    with torch.no_grad():
        reference_coefficients = rasterizer_stage()
        torch.cuda.synchronize(device)
    timings["decode"] = _benchmark_cuda_callable(
        lambda: decode_stage(reference_coefficients),
        args.warmup,
        args.repeat,
        device,
    )

    with torch.no_grad():
        reference_features = decode_stage(reference_coefficients)
        torch.cuda.synchronize(device)
    timings["relevancy_postprocess"] = _benchmark_cuda_callable(
        lambda: postprocess_stage(reference_features),
        args.warmup,
        args.repeat,
        device,
    )
    timings["render_decode"] = _benchmark_cuda_callable(
        lambda: decode_stage(rasterizer_stage()),
        args.warmup,
        args.repeat,
        device,
    )
    timings["total"] = _benchmark_cuda_callable(
        lambda: postprocess_stage(decode_stage(rasterizer_stage())),
        args.warmup,
        args.repeat,
        device,
    )
    benchmark_wall_seconds = float(time.perf_counter() - benchmark_start)

    with torch.no_grad():
        selected_mask, selected_level, selected_peak = postprocess_stage(
            reference_features
        )
        torch.cuda.synchronize(device)
        query_result = {
            "selected_semantic_level_zero_based": int(selected_level.item()),
            "selected_mask_positive_pixels": int(selected_mask.sum().item()),
            "selected_level_peak_relevancy": float(selected_peak.item()),
        }

    properties = torch.cuda.get_device_properties(device)
    geometry_fields = [
        combined_gaussians._xyz,
        combined_gaussians._features_dc,
        combined_gaussians._features_rest,
        combined_gaussians._scaling,
        combined_gaussians._rotation,
        combined_gaussians._opacity,
    ]
    model_counts = {
        "gaussian_points": int(combined_gaussians.get_xyz.shape[0]),
        "geometry_and_rgb_parameter_elements": int(
            sum(tensor.numel() for tensor in geometry_fields)
        ),
        "source_language_logit_elements_three_levels": int(source_logit_elements),
        "combined_codebook_elements": int(
            combined_gaussians._language_feature_codebooks.numel()
        ),
        "combined_sparse_weight_elements": int(
            combined_gaussians._language_feature_weights.numel()
        ),
        "combined_sparse_index_elements": int(
            combined_gaussians._language_feature_indices.numel()
        ),
    }

    result = {
        "schema_version": 1,
        "benchmark_name": "LangSplatV2 released-quick-render query speed",
        "benchmark_classification": [
            "third_party_benchmark",
            "inference_only",
        ],
        "official_benchmark_script": False,
        "started_at": started_at,
        "completed_at": _timestamp_now(),
        "timing_method": {
            "clock": "torch.cuda.Event",
            "synchronization": "end_event.synchronize_per_sample",
            "warmup_per_stage": int(args.warmup),
            "repetitions_per_stage": int(args.repeat),
            "timed_unit": "one camera and one already-encoded positive query",
            "excluded": [
                "Scene construction",
                "image and COLMAP I/O",
                "checkpoint I/O and restore",
                "three-level model merge and top-k extraction",
                "OpenCLIP model loading",
                "positive and negative text encoding",
                "ground-truth loading and metric computation",
                "JSON serialization",
            ],
            "setup_wall_seconds": setup_wall_seconds,
            "benchmark_loop_wall_seconds": benchmark_wall_seconds,
        },
        "stage_definitions": {
            "rasterizer": "released quick sparse-coefficient Gaussian rasterizer, including its RGB output work",
            "decode": "released 3x(64-to-512) einsum and per-pixel L2 normalization",
            "relevancy_postprocess": "OpenCLIP relevancy plus released LERF segmentation postprocess, excluding GT IoU",
            "render_decode": "rasterizer and decode in one CUDA-event interval",
            "total": "rasterizer, decode, relevancy, and released segmentation postprocess in one CUDA-event interval",
        },
        "protocol_notes": {
            "paper_protocol_fully_public": False,
            "postprocess_source": "released eval_lerf.py segmentation_process_cuda operations without GT IoU",
            "paper_code_discrepancy": "The paper appendix describes Avg(M), while released LERF segmentation code uses 0.5*(Avg(M)+M) and an additional 7x7 binary-mask smoother; this benchmark follows released code.",
            "topk_extraction": "released utils.vq_utils.get_weights_and_indices with a device-placement compatibility fix",
            "scene_runtime_files": "Scene input.ply/cameras.json were emitted only in a temporary directory",
        },
        "query": {
            "positive": args.positive,
            "negative_phrases": list(clip_model.negatives),
            "mask_threshold": float(args.mask_threshold),
            "result_outside_timing": query_result,
        },
        "scene": {
            "name": args.scene_name,
            "source_path": str(args.source_path),
            "camera_index_zero_based": int(args.camera_index),
            "camera_name": view.image_name,
            "camera_count": int(camera_count),
            "resolution": {
                "width": int(view.image_width),
                "height": int(view.image_height),
            },
            "white_background": bool(dataset.white_background),
        },
        "model": {
            "model_index": str(args.model_index),
            "requested_checkpoint_iteration": int(args.checkpoint_iteration),
            "checkpoint_root": str(args.checkpoint_root),
            "checkpoints": checkpoint_records,
            "point_counts_per_level": point_counts,
            "counts": model_counts,
            "parameters": {
                "semantic_levels": SEMANTIC_LEVELS,
                "feature_dimension_per_level": FEATURE_DIMENSION,
                "codebook_size_per_level": CODEBOOK_SIZE,
                "topk_per_level": TOPK,
                "sparse_render_channels": SPARSE_RENDER_CHANNELS,
                "coefficient_map_channels": COEFFICIENT_CHANNELS,
                "decoded_feature_channels": SEMANTIC_LEVELS * FEATURE_DIMENSION,
                "spherical_harmonic_degree": int(dataset.sh_degree),
            },
        },
        "software": {
            "repo": _git_metadata(repo_path),
            "python": os.sys.version,
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cudnn": int(torch.backends.cudnn.version())
            if torch.backends.cudnn.version() is not None
            else None,
        },
        "hardware": {
            "device_index_within_visible_set": int(args.device_index),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": properties.name,
            "compute_capability": [int(properties.major), int(properties.minor)],
            "total_memory_bytes": int(properties.total_memory),
            "multiprocessor_count": int(properties.multi_processor_count),
            "driver_version": _driver_version(),
        },
        "paper_reference_a100": {
            "rendering_ms": 2.0,
            "decoding_ms": 0.1,
            "postprocessing_ms": 0.5,
            "total_query_ms": 2.6,
            "feature_render_decode_fps": 476.2,
            "query_fps": 384.6,
            "comparison_warning": "Paper values use one A100 and an incompletely released timing protocol; compare hardware-specific results with this caveat.",
        },
        "timings": timings,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    print(json.dumps(result, indent=2, sort_keys=True))
    print("Wrote {}".format(args.output_json))


if __name__ == "__main__":
    main()
