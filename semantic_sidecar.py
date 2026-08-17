"""Export deployment-only semantic data without duplicating RGB geometry.

The FCGS comparison stores RGB geometry in the FCGS scene bitstream.  A joint
LangSplat checkpoint repeats that geometry (and also contains optimizer state),
so its file size is not a meaningful deployment size.  This module writes only
the sparse semantic weights/indices and the scene-specific semantic codebooks.
It also binds the sidecar to the decoded FCGS point order with an XYZ digest.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from torch import nn


FORMAT_NAME = "fcgs.semantic-sidecar"
FORMAT_VERSION = 1
GEOMETRY_FIELDS = (
    "xyz",
    "features_dc",
    "features_rest",
    "scaling",
    "rotation",
    "opacity",
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    # Hash in rows to avoid materializing a second full-size tensor byte string.
    if tensor.ndim == 0:
        digest.update(tensor.numpy().tobytes())
    else:
        rows_per_chunk = max(1, (8 * 1024 * 1024) // max(1, tensor[0].numel() * tensor.element_size()))
        for start in range(0, tensor.shape[0], rows_per_chunk):
            digest.update(tensor[start:start + rows_per_chunk].numpy().tobytes())
    return digest.hexdigest()


def _tensor_nbytes(tensor):
    return int(tensor.numel() * tensor.element_size())


def _load_checkpoint(path):
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise ValueError("checkpoint must contain (model_params, iteration)")
    return payload[0], int(payload[1])


def _geometry_tensors(model_params):
    if len(model_params) not in (12, 14):
        raise ValueError("expected a 12-field RGB or 14-field joint checkpoint")
    return dict(zip(GEOMETRY_FIELDS, model_params[1:7]))


def _load_ply_xyz(path):
    import numpy as np
    from plyfile import PlyData

    vertices = PlyData.read(path).elements[0]
    xyz = np.stack(
        (np.asarray(vertices["x"]), np.asarray(vertices["y"]), np.asarray(vertices["z"])),
        axis=1,
    ).astype(np.float32, copy=False)
    return torch.from_numpy(xyz.copy())


def _semantic_topk(logits, topk, chunk_size=262144):
    """Match quick-render top-k semantics and return compact CPU tensors."""
    if logits.ndim == 2:
        logits = logits.unsqueeze(1)
    if logits.ndim != 3:
        raise ValueError("semantic logits must be [N,L,K]")
    point_count, level_count, codebook_size = logits.shape
    if topk <= 0 or topk > codebook_size:
        raise ValueError("topk must be in [1, codebook size]")
    if level_count * codebook_size > 256:
        raise ValueError("global semantic indices do not fit uint8")

    weights_parts = []
    indices_parts = []
    offsets = (
        torch.arange(level_count, dtype=torch.int64)
        .view(1, level_count, 1) * codebook_size
    )
    for start in range(0, point_count, chunk_size):
        chunk = logits[start:start + chunk_size].detach().cpu().float()
        probabilities = torch.softmax(chunk, dim=-1)
        weights, indices = torch.topk(probabilities, topk, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-10)
        indices = indices.to(torch.int64) + offsets
        weights_parts.append(weights.reshape(chunk.shape[0], -1).to(torch.float16))
        indices_parts.append(indices.reshape(chunk.shape[0], -1).to(torch.uint8))
    return torch.cat(weights_parts, dim=0), torch.cat(indices_parts, dim=0)


def _verify_geometry(joint_params, geometry_params):
    joint_geometry = _geometry_tensors(joint_params)
    decoded_geometry = _geometry_tensors(geometry_params)
    exact = {}
    for name in GEOMETRY_FIELDS:
        left = joint_geometry[name].detach().cpu()
        right = decoded_geometry[name].detach().cpu()
        exact[name] = bool(left.shape == right.shape and torch.equal(left, right))
    if not all(exact.values()):
        mismatched = [name for name, matches in exact.items() if not matches]
        raise ValueError(
            "semantic-only checkpoint changed or reordered decoded FCGS geometry: {}"
            .format(", ".join(mismatched))
        )
    return exact


def export_sidecar(checkpoint_path, geometry_checkpoint_path, output_path,
                   topk=4, force=False):
    output_path = os.path.abspath(output_path)
    manifest_path = output_path + ".manifest.json"
    if (os.path.exists(output_path) or os.path.exists(manifest_path)) and not force:
        raise FileExistsError("refusing to overwrite existing semantic sidecar")

    joint_params, iteration = _load_checkpoint(checkpoint_path)
    if len(joint_params) != 14:
        raise ValueError("semantic source must be a 14-field joint checkpoint")
    geometry_params, geometry_iteration = _load_checkpoint(geometry_checkpoint_path)
    geometry_exact = _verify_geometry(joint_params, geometry_params)

    semantic_logits = joint_params[7]
    semantic_codebooks = joint_params[8]
    if semantic_logits.ndim == 2:
        semantic_logits = semantic_logits.unsqueeze(1)
    if semantic_codebooks.ndim == 3:
        semantic_codebooks = semantic_codebooks.unsqueeze(0)
    if semantic_logits.ndim != 3 or semantic_codebooks.ndim != 4:
        raise ValueError("unsupported semantic tensor shapes")

    level_count, rvq_layers, codebook_size, feature_dim = semantic_codebooks.shape
    if rvq_layers != 1:
        raise ValueError("released quick renderer requires exactly one RVQ layer")
    if semantic_logits.shape[1:] != (level_count, rvq_layers * codebook_size):
        raise ValueError("semantic logits and codebooks disagree")
    if level_count != 3 or codebook_size != 64 or feature_dim != 512:
        raise ValueError("released quick renderer requires [3,1,64,512] codebooks")

    semantic_weights, semantic_indices = _semantic_topk(semantic_logits, topk)
    # Strip the singleton RVQ dimension to match GaussianModel quick rendering.
    semantic_codebooks = semantic_codebooks[:, 0].detach().cpu().contiguous().to(torch.float16)
    xyz = _geometry_tensors(geometry_params)["xyz"]
    point_count = int(xyz.shape[0])
    if semantic_weights.shape[0] != point_count:
        raise ValueError("semantic rows do not match decoded FCGS point count")

    tensors = {
        "semantic_weights": semantic_weights.contiguous(),
        "semantic_indices": semantic_indices.contiguous(),
        "semantic_codebooks": semantic_codebooks,
    }
    bundle = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "iteration": iteration,
        "geometry_iteration": geometry_iteration,
        "point_count": point_count,
        "topk": int(topk),
        "semantic_level_count": int(level_count),
        "codebook_size": int(codebook_size),
        "feature_dim": int(feature_dim),
        "geometry_xyz_sha256": _sha256_tensor(xyz),
        "tensors": tensors,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_path)

    logical_bytes = {name: _tensor_nbytes(value) for name, value in tensors.items()}
    manifest = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "semantic_sidecar": os.path.basename(output_path),
        "semantic_sidecar_bytes": int(os.path.getsize(output_path)),
        "shared_semantic_bytes": 0,
        "shared_semantic_components": [],
        "sha256": _sha256_file(output_path),
        "point_count": point_count,
        "topk": int(topk),
        "geometry_xyz_sha256": bundle["geometry_xyz_sha256"],
        "geometry_exact_match": geometry_exact,
        "contains_geometry": False,
        "logical_tensor_bytes": logical_bytes,
        "logical_tensor_bytes_total": int(sum(logical_bytes.values())),
        "tensor_schema": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in tensors.items()
        },
        "excluded_training_state": [
            "geometry", "rgb_attributes", "opacity", "optimizer",
            "dense_semantic_logits", "gradient_accumulators",
        ],
        "source": {
            "joint_checkpoint": os.path.abspath(checkpoint_path),
            "geometry_checkpoint": os.path.abspath(geometry_checkpoint_path),
            "iteration": iteration,
        },
    }
    with open(manifest_path, "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
    return manifest


def load_sidecar_into_gaussians(gaussians, sidecar_path, device=None):
    """Attach sparse semantic tensors to an already loaded RGB Gaussian model."""
    bundle = torch.load(sidecar_path, map_location="cpu")
    if bundle.get("format") != FORMAT_NAME or bundle.get("version") != FORMAT_VERSION:
        raise ValueError("unsupported semantic sidecar format")
    tensors = bundle["tensors"]
    if any(name in tensors for name in GEOMETRY_FIELDS):
        raise ValueError("semantic sidecar unexpectedly contains geometry")
    if int(gaussians._xyz.shape[0]) != int(bundle["point_count"]):
        raise ValueError("semantic sidecar and RGB geometry have different point counts")
    if _sha256_tensor(gaussians._xyz) != bundle["geometry_xyz_sha256"]:
        raise ValueError("semantic sidecar does not match the RGB point ordering")

    if device is None:
        device = gaussians._xyz.device
    weights = tensors["semantic_weights"]
    indices = tensors["semantic_indices"]
    codebooks = tensors["semantic_codebooks"]
    expected_width = int(bundle["semantic_level_count"]) * int(bundle["topk"])
    if weights.shape != indices.shape or weights.shape != (bundle["point_count"], expected_width):
        raise ValueError("invalid sparse semantic tensor shape")
    if codebooks.shape != (
        bundle["semantic_level_count"], bundle["codebook_size"], bundle["feature_dim"]
    ):
        raise ValueError("invalid semantic codebook shape")

    gaussians._language_feature_weights = weights.float().to(device)
    # The released CUDA rasterizer consumes floating-point indices.
    gaussians._language_feature_indices = indices.float().to(device)
    gaussians._language_feature_codebooks = nn.Parameter(
        codebooks.float().to(device), requires_grad=False
    )
    return gaussians, bundle


def validate_sidecar(sidecar_path, geometry_checkpoint_path=None,
                     source_checkpoint_path=None, geometry_ply_path=None):
    from scene.gaussian_model import GaussianModel

    gaussians = GaussianModel(3)
    if geometry_checkpoint_path:
        geometry_params, _ = _load_checkpoint(geometry_checkpoint_path)
        # Test-mode restore does not construct or load optimizer state.
        class _Args:
            # GaussianModel's legacy 12-field restore skips optimizer loading
            # when include_feature is true; semantics are attached below.
            include_feature = True
        gaussians.restore(geometry_params, _Args(), mode="test")
    elif geometry_ply_path:
        # CPU validation needs only the point count/order that binds the
        # sidecar.  End-to-end evaluation loads all PLY fields on CUDA.
        gaussians._xyz = _load_ply_xyz(geometry_ply_path)
    else:
        raise ValueError("provide a geometry checkpoint or decoded FCGS PLY")
    gaussians, bundle = load_sidecar_into_gaussians(
        gaussians, sidecar_path, device="cpu"
    )
    result = {
        "reload_ok": True,
        "geometry_match": True,
        "contains_geometry": False,
        "point_count": int(bundle["point_count"]),
        "semantic_sidecar_bytes": int(os.path.getsize(sidecar_path)),
        "sha256": _sha256_file(sidecar_path),
    }

    if source_checkpoint_path:
        source_params, _ = _load_checkpoint(source_checkpoint_path)
        if len(source_params) != 14:
            raise ValueError("source checkpoint must contain semantic tensors")
        reference_weights, reference_indices = _semantic_topk(
            source_params[7], int(bundle["topk"])
        )
        source_codebooks = source_params[8]
        if source_codebooks.ndim == 4:
            source_codebooks = source_codebooks[:, 0]
        loaded_weights = gaussians._language_feature_weights.cpu()
        loaded_indices = gaussians._language_feature_indices.cpu().to(torch.uint8)
        loaded_codebooks = gaussians._language_feature_codebooks.detach().cpu()
        result.update({
            "indices_exact": bool(torch.equal(reference_indices, loaded_indices)),
            "weights_max_abs_error": float(
                (reference_weights.float() - loaded_weights).abs().max().item()
            ),
            "codebooks_max_abs_error": float(
                (source_codebooks.float() - loaded_codebooks).abs().max().item()
            ),
        })
        if not result["indices_exact"]:
            raise ValueError("reloaded semantic indices differ from source")

        # Compare actual reconstructed semantic vectors on a deterministic
        # subset.  This tests the index/weight/codebook composition used by
        # the renderer, not only serialization of its individual tensors.
        logits = source_params[7]
        if logits.ndim == 2:
            logits = logits.unsqueeze(1)
        sample_count = min(4096, logits.shape[0])
        sample_rows = torch.linspace(
            0, logits.shape[0] - 1, steps=sample_count
        ).round().to(torch.int64)
        sample_logits = logits[sample_rows].float()
        probabilities = torch.softmax(sample_logits, dim=-1)
        source_weights, source_indices = torch.topk(
            probabilities, int(bundle["topk"]), dim=-1
        )
        source_weights = source_weights / (
            source_weights.sum(dim=-1, keepdim=True) + 1e-10
        )
        source_codebooks = source_codebooks.float()
        level_grid = torch.arange(source_codebooks.shape[0]).view(1, -1, 1)
        source_features = (
            source_weights.unsqueeze(-1)
            * source_codebooks[level_grid, source_indices]
        ).sum(dim=2)

        loaded_weights_sample = loaded_weights[sample_rows].view(
            sample_count, bundle["semantic_level_count"], bundle["topk"]
        )
        loaded_indices_sample = loaded_indices[sample_rows].to(torch.int64).view(
            sample_count, bundle["semantic_level_count"], bundle["topk"]
        )
        loaded_codebooks_flat = loaded_codebooks.reshape(-1, loaded_codebooks.shape[-1])
        loaded_features = (
            loaded_weights_sample.unsqueeze(-1)
            * loaded_codebooks_flat[loaded_indices_sample]
        ).sum(dim=2)
        feature_error = (source_features - loaded_features).abs()
        cosine = torch.nn.functional.cosine_similarity(
            source_features.reshape(-1, source_features.shape[-1]),
            loaded_features.reshape(-1, loaded_features.shape[-1]),
            dim=-1,
        )
        result.update({
            "semantic_feature_sample_count": int(sample_count),
            "semantic_feature_max_abs_error": float(feature_error.max().item()),
            "semantic_feature_cosine_mean": float(cosine.mean().item()),
            "semantic_feature_cosine_min": float(cosine.min().item()),
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--geometry-checkpoint", required=True)
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--topk", type=int, default=4)
    export_parser.add_argument("--force", action="store_true")
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--sidecar", required=True)
    geometry_group = validate_parser.add_mutually_exclusive_group(required=True)
    geometry_group.add_argument("--geometry-checkpoint")
    geometry_group.add_argument("--geometry-ply")
    validate_parser.add_argument("--source-checkpoint")
    args = parser.parse_args()

    if args.command == "export":
        result = export_sidecar(
            args.checkpoint, args.geometry_checkpoint, args.output,
            topk=args.topk, force=args.force,
        )
    else:
        result = validate_sidecar(
            args.sidecar, args.geometry_checkpoint, args.source_checkpoint,
            geometry_ply_path=args.geometry_ply,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
