"""Evaluate RGB metrics for a standard 3DGS PLY with the common renderer."""

import argparse
import json
import os

import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel, render
from lpipsPyTorch.modules.lpips import LPIPS
from scene import Scene
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=30_000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset = model.extract(args)
    dataset.eval = True
    pipe = pipeline.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        shuffle=False,
    )
    views = scene.getTestCameras()
    if not views:
        raise RuntimeError("the dataset has no held-out test cameras")

    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    lpips_metric = LPIPS("vgg", "0.1").cuda().eval()
    render_options = argparse.Namespace(include_feature=False, quick_render=False)

    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "l1": 0.0}
    per_view = []
    with torch.no_grad():
        for view in tqdm(views, desc="PLY RGB evaluation"):
            image = render(
                view, gaussians, pipe, background, render_options
            )["render"]
            # Match the standard 3DGS saved-image evaluator and Table 1 RGB
            # protocol: metrics are computed on uint8-quantized renders.
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
        "source": os.path.abspath(dataset.source_path),
        "model_path": os.path.abspath(dataset.model_path),
        "iteration": args.iteration,
        "point_count": int(gaussians.get_xyz.shape[0]),
        "test_views": count,
        "metrics": {name: value / count for name, value in totals.items()},
        "protocol": "common renderer, LLFF holdout=8, uint8 render metrics",
        "per_view": per_view,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = output + ".tmp"
    with open(temporary, "w") as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True)
    os.replace(temporary, output)
    print(json.dumps({key: result[key] for key in ("point_count", "test_views", "metrics")}, indent=2))


if __name__ == "__main__":
    main()
