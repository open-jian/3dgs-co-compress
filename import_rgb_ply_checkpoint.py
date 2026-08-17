"""Create a frozen RGB checkpoint from a trained 3DGS PLY.

The language stage only needs the trained Gaussian tensors; it intentionally
starts with a fresh optimizer.  This utility reconstructs the 12-field
checkpoint schema expected by ``train_joint.py`` without retraining RGB 3DGS.
"""

import argparse
import os

import torch

from scene.dataset_readers import readColmapSceneInfo, readScanNetInfo
from scene.gaussian_model import GaussianModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.output):
        raise FileExistsError("Refusing to overwrite {}".format(args.output))

    if "scannet" in os.path.abspath(args.source).lower():
        scene_info = readScanNetInfo(args.source, False, args.eval)
    else:
        scene_info = readColmapSceneInfo(args.source, args.images, args.eval)
    model = GaussianModel(sh_degree=3)
    model.load_ply(args.ply)
    count = model.get_xyz.shape[0]
    device = model.get_xyz.device

    zeros_1d = torch.zeros(count, dtype=torch.float32, device=device)
    zeros_2d = torch.zeros((count, 1), dtype=torch.float32, device=device)
    state = (
        model.active_sh_degree,
        model._xyz.detach(),
        model._features_dc.detach(),
        model._features_rest.detach(),
        model._scaling.detach(),
        model._rotation.detach(),
        model._opacity.detach(),
        zeros_1d,
        zeros_2d,
        zeros_2d.clone(),
        {},  # RGB optimizer state is deliberately not reused by train_joint.py.
        float(scene_info.nerf_normalization["radius"]),
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save((state, 30_000), args.output)
    print(
        "Saved {} Gaussians (radius {:.6f}) to {}".format(
            count, state[-1], args.output
        )
    )


if __name__ == "__main__":
    main()
