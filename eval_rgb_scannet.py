#!/usr/bin/env python3
"""Small held-out RGB sanity check for a native-ScanNet 3DGS checkpoint."""

import argparse
import json
import time
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


def main():
    parser = argparse.ArgumentParser()
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dataset = model.extract(args)
    dataset.eval = True
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    params, iteration = torch.load(args.checkpoint)
    # In evaluation mode no optimizer is constructed. ``include_feature=True``
    # here only tells the legacy restore path not to load RGB optimizer state;
    # the renderer below remains explicitly RGB-only.
    gaussians.restore(params, SimpleNamespace(include_feature=True), mode="test")
    views = scene.getTestCameras()
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    render_options = SimpleNamespace(include_feature=False, quick_render=False, topk=1)
    records, render_seconds = [], []
    with torch.no_grad():
        for view in tqdm(views, desc="held-out ScanNet RGB"):
            torch.cuda.synchronize()
            started = time.perf_counter()
            image = render(view, gaussians, pipe, background, render_options)["render"]
            torch.cuda.synchronize()
            render_seconds.append(time.perf_counter() - started)
            ground_truth = view.original_image[:3].cuda()
            records.append(
                {
                    "image": view.image_name,
                    "psnr": float(psnr(image, ground_truth).mean()),
                    "ssim": float(ssim(image, ground_truth).mean()),
                    "l1": float(l1_loss(image, ground_truth).mean()),
                }
            )
    result = {
        "scope": "ScanNet public scene0000_00 mini; every eighth mini frame held out",
        "checkpoint": args.checkpoint,
        "iteration": int(iteration),
        "test_views": len(records),
        "metrics": {
            key: float(np.mean([record[key] for record in records]))
            for key in ("psnr", "ssim", "l1")
        },
        "render_ms_per_view": 1000.0 * float(np.mean(render_seconds)),
        "per_view": records,
    }
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({key: result[key] for key in ("test_views", "metrics", "render_ms_per_view")}, indent=2))


if __name__ == "__main__":
    main()
