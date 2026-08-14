"""Export a standard 12-field RGB 3DGS checkpoint as a binary PLY."""

import argparse
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def _field_names(features_dc, features_rest, scaling, rotation):
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names.extend(f"f_dc_{index}" for index in range(features_dc.shape[1] * features_dc.shape[2]))
    names.extend(
        f"f_rest_{index}"
        for index in range(features_rest.shape[1] * features_rest.shape[2])
    )
    names.append("opacity")
    names.extend(f"scale_{index}" for index in range(scaling.shape[1]))
    names.extend(f"rot_{index}" for index in range(rotation.shape[1]))
    return names


def export_checkpoint(checkpoint_path, output_path, overwrite=False):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing PLY: {output_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise ValueError("Expected checkpoint payload (model_state, iteration)")
    state, iteration = payload
    if len(state) != 12:
        raise ValueError(f"Expected a 12-field RGB checkpoint, found {len(state)} fields")

    xyz = state[1].detach().cpu().numpy()
    features_dc = state[2].detach().cpu().numpy()
    features_rest = state[3].detach().cpu().numpy()
    scaling = state[4].detach().cpu().numpy()
    rotation = state[5].detach().cpu().numpy()
    opacity = state[6].detach().cpu().numpy()
    del payload, state

    point_count = xyz.shape[0]
    tensors = (features_dc, features_rest, scaling, rotation, opacity)
    if any(tensor.shape[0] != point_count for tensor in tensors):
        raise ValueError("Checkpoint attributes do not share one Gaussian count")
    if xyz.shape[1] != 3 or opacity.shape[1] != 1:
        raise ValueError("Unexpected xyz or opacity shape")

    names = _field_names(features_dc, features_rest, scaling, rotation)
    vertices = np.empty(point_count, dtype=[(name, "<f4") for name in names])
    for axis, name in enumerate(("x", "y", "z")):
        vertices[name] = xyz[:, axis]
    for name in ("nx", "ny", "nz"):
        vertices[name] = 0.0

    dc_index = 0
    for channel in range(features_dc.shape[2]):
        for coefficient in range(features_dc.shape[1]):
            vertices[f"f_dc_{dc_index}"] = features_dc[:, coefficient, channel]
            dc_index += 1

    rest_index = 0
    for channel in range(features_rest.shape[2]):
        for coefficient in range(features_rest.shape[1]):
            vertices[f"f_rest_{rest_index}"] = features_rest[:, coefficient, channel]
            rest_index += 1

    vertices["opacity"] = opacity[:, 0]
    for index in range(scaling.shape[1]):
        vertices[f"scale_{index}"] = scaling[:, index]
    for index in range(rotation.shape[1]):
        vertices[f"rot_{index}"] = rotation[:, index]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(output_path)
    print(
        f"Exported iteration {iteration} with {point_count} Gaussians: "
        f"{checkpoint_path} -> {output_path}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
