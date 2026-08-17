"""Evaluate RGB quality strictly from a deployment-only ClunoGS artifact."""

import argparse
import json
import os
import time
from types import SimpleNamespace

import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from compact_artifact import load_compact_gaussians
from gaussian_renderer import GaussianModel, render
from lpipsPyTorch.modules.lpips import LPIPS
from scene import Scene
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--compact_artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset = model.extract(args)
    dataset.eval = True
    dataset.model_path = os.path.dirname(os.path.abspath(args.compact_artifact))
    pipe = pipeline.extract(args)

    camera_model = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, camera_model, shuffle=False)
    views = scene.getTestCameras()
    if not views:
        raise RuntimeError("the dataset has no held-out test cameras")

    load_start = time.perf_counter()
    gaussians, bundle = load_compact_gaussians(args.compact_artifact, device="cuda")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    lpips_metric = LPIPS("vgg", "0.1").cuda().eval()
    render_options = SimpleNamespace(include_feature=False, quick_render=False)

    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "l1": 0.0}
    per_view = []
    render_seconds = []
    with torch.no_grad():
        for view in tqdm(views, desc="compact RGB evaluation"):
            torch.cuda.synchronize()
            render_start = time.perf_counter()
            image = render(
                view, gaussians, pipe, background, render_options
            )["render"]
            torch.cuda.synchronize()
            render_seconds.append(time.perf_counter() - render_start)
            # Match the standard 3DGS saved-image evaluator exactly.
            image = torch.round(image.mul(255).clamp_(0, 255)) / 255.0
            ground_truth = view.original_image[:3].cuda()
            values = {
                "psnr": float(psnr(image, ground_truth).mean()),
                "ssim": float(ssim(image, ground_truth).mean()),
                "lpips": float(lpips_metric(image, ground_truth).mean()),
                "l1": float(l1_loss(image, ground_truth).mean()),
            }
            for name, value in values.items():
                totals[name] += value
            per_view.append({"image": view.image_name, **values})

    count = len(views)
    result = {
        "artifact": os.path.abspath(args.compact_artifact),
        "point_count": int(bundle["point_count"]),
        "test_views": count,
        "metrics": {name: value / count for name, value in totals.items()},
        "timing": {
            "representation_load_seconds": load_seconds,
            "rgb_render_ms_per_view": 1000.0 * sum(render_seconds) / count,
        },
        "per_view": per_view,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True)
    print(json.dumps({key: result[key] for key in ("point_count", "test_views", "metrics", "timing")}, indent=2))


if __name__ == "__main__":
    main()
