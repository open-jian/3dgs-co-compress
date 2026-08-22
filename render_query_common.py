#!/usr/bin/env python3
"""Evaluate a fresh-reloaded ClunoGS compact artifact on common 3D-OVS."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from compact_artifact import load_compact_gaussians
from eval.openclip_encoder import OpenCLIPNetwork
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.image_utils import psnr
from utils.general_utils import safe_state


SEMANTIC_LEVELS = 3
CODEBOOK_SIZE = 64


def load_annotations(dataset: Path):
    root = dataset / "segmentations"
    classes = [
        line.strip()
        for line in (root / "classes.txt").read_text().splitlines()
        if line.strip()
    ]
    annotations = {}
    for view_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        annotations[view_dir.name] = {
            prompt: cv2.imread(
                str(view_dir / f"{prompt}.png"), cv2.IMREAD_GRAYSCALE
            ) > 0
            for prompt in classes
        }
    if not annotations:
        raise FileNotFoundError(f"No 3D-OVS annotations found in {root}")
    return annotations


@torch.no_grad()
def render_coefficients(view, gaussians, pipeline, background, render_options):
    package = render(view, gaussians, pipeline, background, render_options)
    coefficients = package["language_feature_weight_map"]
    _, height, width = coefficients.shape
    coefficients = coefficients.view(
        SEMANTIC_LEVELS, CODEBOOK_SIZE, height, width
    )
    return package["render"], coefficients


def relevance_from_coefficients(
    coefficients, codebooks, positive, negatives, pixel_chunk_size,
):
    """Exactly reconstruct and score one prompt in bounded pixel chunks."""
    levels, _, height, width = coefficients.shape
    pixel_count = height * width
    relevance = torch.empty(
        (levels, height, width), dtype=coefficients.dtype,
        device=coefficients.device,
    )
    phrases = torch.cat((positive, negatives), dim=0).to(
        device=coefficients.device, dtype=coefficients.dtype
    )
    for level in range(levels):
        weights = coefficients[level].view(CODEBOOK_SIZE, pixel_count)
        dictionary = codebooks[level].to(coefficients.dtype)
        for start in range(0, pixel_count, pixel_chunk_size):
            stop = min(start + pixel_chunk_size, pixel_count)
            features = torch.einsum(
                "dk,kn->dn", dictionary.transpose(0, 1), weights[:, start:stop]
            )
            features = features / features.norm(dim=0, keepdim=True).clamp_min_(1e-10)
            similarities = torch.einsum(
                "qc,pc->qp", features.transpose(0, 1).contiguous(), phrases
            )
            positive_similarity = similarities[:, :1]
            negative_similarity = similarities[:, 1:]
            pair_logits = torch.stack(
                (positive_similarity.expand_as(negative_similarity), negative_similarity),
                dim=-1,
            )
            relevance[level].view(-1)[start:stop] = torch.softmax(
                10.0 * pair_logits, dim=-1
            )[..., 0].min(dim=1).values
    return relevance


def threshold_mask(heatmap: torch.Tensor, threshold: float):
    averaged = torch.nn.functional.avg_pool2d(
        heatmap[None, None], 29, stride=1, padding=14,
        count_include_pad=False,
    )[0, 0]
    blended = 0.5 * (averaged + heatmap)
    normalized = blended - blended.min()
    normalized = normalized / (normalized.max() + 1e-9)
    normalized = torch.clamp(normalized * 2.0 - 1.0, 0.0, 1.0)
    return normalized > threshold, blended


def processed_mask(heatmap: torch.Tensor, threshold: float):
    thresholded, blended = threshold_mask(heatmap, threshold)
    smoothed = torch.nn.functional.avg_pool2d(
        thresholded.float()[None, None], 7, stride=1, padding=3,
        count_include_pad=False,
    )[0, 0]
    return smoothed > 0.5, blended


def select_level_mask(valid_map, prompt_index, threshold):
    masks = []
    heats = []
    for level in range(valid_map.shape[0]):
        mask, heat = processed_mask(valid_map[level, prompt_index], threshold)
        masks.append(mask)
        heats.append(heat)

    # Match the released 3D-OVS evaluator: level zero is the fallback and the
    # score comparison is performed over semantic levels one and two.
    scores = torch.zeros(len(masks), device=valid_map.device)
    for level in range(1, len(masks)):
        denominator = masks[level].sum()
        if denominator:
            scores[level] = (heats[level] * masks[level]).sum() / denominator
    selected = int(torch.argmax(scores).item())
    return masks[selected], selected


def set_cached_text(clip_model, prompts, text_cache):
    key = tuple(prompts)
    clip_model.positives = list(prompts)
    clip_model.pos_embeds = text_cache[key]


@torch.no_grad()
def one_query(
    view,
    prompt,
    scene_name,
    gaussians,
    pipeline,
    background,
    render_options,
    clip_model,
    text_cache,
    threshold,
    pixel_chunk_size,
):
    _, coefficients = render_coefficients(
        view, gaussians, pipeline, background, render_options
    )
    positive = text_cache[(prompt,)]
    codebooks = gaussians.get_quick_codebooks()
    relevance = relevance_from_coefficients(
        coefficients, codebooks, positive, clip_model.neg_embeds,
        pixel_chunk_size,
    )
    masks = []
    heats = []
    for level in range(SEMANTIC_LEVELS):
        mask, heat = processed_mask(relevance[level], threshold)
        masks.append(mask)
        heats.append(heat)
    scores = torch.zeros(SEMANTIC_LEVELS, device=relevance.device)
    for level in range(1, SEMANTIC_LEVELS):
        denominator = masks[level].sum()
        if denominator:
            scores[level] = (heats[level] * masks[level]).sum() / denominator
    return masks[int(torch.argmax(scores).item())]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_group = ModelParams(parser, sentinel=False)
    pipeline_group = PipelineParams(parser)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--method", default="ClunoGS")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--compact_artifact", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--mask_thresh", type=float, default=0.4)
    parser.add_argument("--latency_warmup", type=int, default=3)
    parser.add_argument("--latency_repeats", type=int, default=1)
    parser.add_argument("--include_feature", action="store_true")
    parser.add_argument("--quick_render", action="store_true")
    parser.add_argument("--relevance_pixel_chunk", type=int, default=8192)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    args.include_feature = True
    args.quick_render = True
    safe_state(args.quiet)

    started = time.time()
    dataset = model_group.extract(args)
    dataset.eval = True
    dataset.model_path = str(args.output_dir)
    pipeline = pipeline_group.extract(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    camera_holder = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, camera_holder, shuffle=False)
    cameras = {
        view.image_name.split(".")[0]: view
        for view in scene.getTrainCameras() + scene.getTestCameras()
    }

    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    gaussians, bundle = load_compact_gaussians(
        args.compact_artifact, device="cuda"
    )
    clip_model = OpenCLIPNetwork(torch.device("cuda"))
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    annotations = load_annotations(args.dataset)
    missing = sorted(set(annotations) - set(cameras))
    if missing:
        raise KeyError(f"Missing calibrated annotated cameras: {missing}")

    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    render_options = SimpleNamespace(include_feature=True, quick_render=True)
    codebooks = gaussians.get_quick_codebooks()

    pairs = []
    psnrs = []
    for view_name, view_ann in tqdm(
        annotations.items(), desc="3D-OVS annotated views"
    ):
        view = cameras[view_name]
        prompts = list(view_ann)
        clip_model.set_positives(prompts)
        rgb, coefficients = render_coefficients(
            view, gaussians, pipeline, background, render_options
        )
        rgb = torch.round(rgb.mul(255).clamp_(0, 255)) / 255.0
        psnrs.append(float(psnr(rgb, view.original_image[:3].cuda()).mean()))
        relevances = []
        for prompt in prompts:
            clip_model.set_positives([prompt])
            relevances.append(relevance_from_coefficients(
                coefficients, codebooks, clip_model.pos_embeds,
                clip_model.neg_embeds, args.relevance_pixel_chunk,
            ))

        view_output = args.output_dir / view_name
        view_output.mkdir(parents=True, exist_ok=True)
        for prompt_index, prompt in enumerate(prompts):
            valid_map = relevances[prompt_index][:, None]
            predicted, selected_level = select_level_mask(valid_map, 0, args.mask_thresh)
            target_np = view_ann[prompt]
            if tuple(target_np.shape) != tuple(predicted.shape):
                target_np = cv2.resize(
                    target_np.astype(np.uint8),
                    (predicted.shape[1], predicted.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            target = torch.from_numpy(target_np).to(predicted.device)
            intersection = int(torch.logical_and(predicted, target).sum().item())
            union = int(torch.logical_or(predicted, target).sum().item())
            positives = int(target.sum().item())
            pairs.append({
                "view": view_name,
                "prompt": prompt,
                "iou": intersection / union,
                "accuracy": intersection / positives,
                "selected_level": selected_level,
            })
            cv2.imwrite(
                str(view_output / f"chosen_{prompt}.png"),
                predicted.cpu().numpy().astype(np.uint8) * 255,
            )
        del coefficients, relevances, rgb

    query_set = [
        (cameras[view_name], prompt)
        for view_name, view_ann in annotations.items()
        for prompt in view_ann
    ]
    text_cache = {}
    for _, prompt in query_set:
        prompts = [prompt]
        key = tuple(prompts)
        if key not in text_cache:
            clip_model.set_positives(prompts)
            text_cache[key] = clip_model.pos_embeds.detach().clone()

    for index in range(args.latency_warmup):
        view, prompt = query_set[index % len(query_set)]
        one_query(
            view, prompt, args.scene, gaussians, pipeline, background,
            render_options, clip_model, text_cache, args.mask_thresh,
            args.relevance_pixel_chunk,
        )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    repeat_seconds = []
    for _ in range(args.latency_repeats):
        torch.cuda.synchronize()
        query_started = time.perf_counter()
        for view, prompt in query_set:
            one_query(
                view, prompt, args.scene, gaussians, pipeline, background,
                render_options, clip_model, text_cache, args.mask_thresh,
                args.relevance_pixel_chunk,
            )
        torch.cuda.synchronize()
        repeat_seconds.append(time.perf_counter() - query_started)

    result = {
        "method": args.method,
        "scene": args.scene,
        "artifact": str(Path(args.compact_artifact).resolve()),
        "artifact_bytes": int(os.path.getsize(args.compact_artifact)),
        "gaussian_count": int(bundle["point_count"]),
        "miou": float(np.mean([pair["iou"] for pair in pairs])),
        "macc": float(np.mean([pair["accuracy"] for pair in pairs])),
        "psnr": float(np.mean(psnrs)),
        "annotated_views": len(annotations),
        "num_view_prompt_pairs": len(pairs),
        "model_load_seconds": load_seconds,
        "query_latency_definition": (
            "resident compact semantic reconstruction, cached-text scoring, "
            "mask postprocessing, and level selection"
        ),
        "query_latency_ms": (
            1000.0 * sum(repeat_seconds)
            / (len(query_set) * len(repeat_seconds))
        ),
        "relevance_pixel_chunk": args.relevance_pixel_chunk,
        "query_memory_path": "3x64 coefficient render plus exact chunked 512-D reconstruction",
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "wall_seconds": time.time() - started,
        "per_view_psnr": psnrs,
        "pairs": pairs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.output_json)
    print(json.dumps({
        key: result[key]
        for key in (
            "miou", "macc", "psnr", "artifact_bytes",
            "query_latency_ms", "peak_allocated_bytes",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
