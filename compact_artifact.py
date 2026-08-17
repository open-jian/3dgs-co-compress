"""Export and load the deployment-only ClunoGS scene artifact.

The training checkpoint intentionally contains optimizer state, dense semantic
logits, and unquantized SH tensors.  Those objects are not deployment payloads
and must never be used for size or quality evaluation.  This module converts a
finished joint checkpoint plus its SH assignment file into the only artifact
consumed by evaluation.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


FORMAT_NAME = "clunogs.compact-scene"
FORMAT_VERSION = 1


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_nbytes(tensor):
    return int(tensor.numel() * tensor.element_size())


def _sha256_tensor(tensor):
    """Hash tensor values in contiguous row order, independent of torch.save."""
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    if tensor.ndim == 0:
        digest.update(tensor.numpy().tobytes())
    else:
        row_bytes = max(1, tensor[0].numel() * tensor.element_size())
        rows_per_chunk = max(1, (8 * 1024 * 1024) // row_bytes)
        for start in range(0, tensor.shape[0], rows_per_chunk):
            digest.update(tensor[start:start + rows_per_chunk].numpy().tobytes())
    return digest.hexdigest()


def _semantic_topk(logits, topk, chunk_size=262144):
    """Return sparse normalized weights and global dictionary indices on CPU."""
    if logits.ndim == 2:
        logits = logits.unsqueeze(1)
    if logits.ndim != 3:
        raise ValueError("semantic logits must be [N,L,K]")
    point_count, level_count, codebook_size = logits.shape
    if topk <= 0 or topk > codebook_size:
        raise ValueError("topk must be in [1, codebook size]")
    if level_count * codebook_size > 256:
        raise ValueError("global semantic indices do not fit uint8")

    all_weights = []
    all_indices = []
    offsets = torch.arange(level_count, dtype=torch.int64).view(1, level_count, 1)
    offsets = offsets * codebook_size
    for start in range(0, point_count, chunk_size):
        chunk = logits[start:start + chunk_size].float()
        probabilities = torch.softmax(chunk, dim=-1)
        weights, indices = torch.topk(probabilities, topk, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-10)
        indices = indices.to(torch.int64) + offsets
        all_weights.append(weights.reshape(chunk.shape[0], -1).to(torch.float16))
        all_indices.append(indices.reshape(chunk.shape[0], -1).to(torch.uint8))
    return torch.cat(all_weights, dim=0), torch.cat(all_indices, dim=0)


def export_compact_artifact(checkpoint_path, sh_quantization_path,
                            output_path, topk=4, force=False,
                            source_id_sidecar_path=None):
    output_path = os.path.abspath(output_path)
    manifest_path = output_path + ".manifest.json"
    if (os.path.exists(output_path) or os.path.exists(manifest_path)) and not force:
        raise FileExistsError("refusing to overwrite existing compact artifact")

    model_params, iteration = torch.load(checkpoint_path, map_location="cpu")
    if len(model_params) != 14:
        raise ValueError("expected a 14-field joint checkpoint")
    (active_sh_degree, xyz, features_dc, features_rest, scaling, rotation,
     opacity, semantic_logits, semantic_codebooks, _max_radii2d,
     _xyz_gradient_accum, _denom, _optimizer_state,
     _spatial_lr_scale) = model_params

    point_count = int(xyz.shape[0])
    source_id_lineage = None
    embedded_source_ids = None
    if source_id_sidecar_path:
        from source_id_lineage import load_lineage_sidecar

        source_ids, source_bundle = load_lineage_sidecar(
            source_id_sidecar_path, checkpoint_path=checkpoint_path
        )
        if source_ids.shape[0] != point_count:
            raise ValueError("source-ID sidecar and compact rows disagree")
        embedded_source_ids = source_ids.detach().cpu().to(torch.int64).contiguous()
        source_id_lineage = {
            "format": source_bundle["format"],
            "version": int(source_bundle["version"]),
            "mapping": source_bundle["mapping"],
            "origin_point_count": int(source_bundle["origin_point_count"]),
            "origin_checkpoint_sha256": source_bundle[
                "origin_checkpoint_sha256"
            ],
            "origin_xyz_sha256": source_bundle["origin_xyz_sha256"],
            "remaining_point_count": int(source_bundle["remaining_point_count"]),
            "deleted_source_point_count": int(
                source_bundle["deleted_source_point_count"]
            ),
            "bound_checkpoint_sha256": source_bundle[
                "bound_checkpoint_sha256"
            ],
            "bound_xyz_sha256": source_bundle["bound_xyz_sha256"],
            "remaining_source_ids_sha256": _sha256_tensor(embedded_source_ids),
            "embedded_tensor": "source_ids",
            "embedded_dtype": "torch.int64",
            "embedded_logical_bytes": _tensor_nbytes(embedded_source_ids),
            "embedded_in_deployment_artifact": True,
            "counted_in_deployment_bytes": True,
        }
    expected_feature_shape = tuple(features_rest.shape[1:])
    if sh_quantization_path:
        sh_state = torch.load(sh_quantization_path, map_location="cpu")
        sh_centers = sh_state["centers"].contiguous().float()
        sh_indices_long = sh_state["indices"].reshape(-1).to(torch.int64)
        if sh_indices_long.shape[0] != point_count:
            raise ValueError("SH assignment count does not match Gaussian count")
        if sh_centers.shape[0] > 256 or int(sh_indices_long.max()) >= 256:
            raise ValueError("SH assignments do not fit uint8")
        stored_feature_shape = tuple(sh_state["feature_shape"])
        if stored_feature_shape != expected_feature_shape:
            raise ValueError("SH feature shape does not match checkpoint")
        sh_tensors = {
            "sh_codebook": sh_centers,
            "sh_indices": sh_indices_long.to(torch.uint8),
        }
        sh_mode = "vq_uint8"
    else:
        stored_feature_shape = expected_feature_shape
        sh_tensors = {
            "features_rest": features_rest.detach().cpu().contiguous().float(),
        }
        sh_mode = "full_float32"

    semantic_weights, semantic_indices = _semantic_topk(
        semantic_logits.detach().cpu(), topk
    )
    semantic_codebooks = semantic_codebooks.detach().cpu().contiguous().to(torch.float16)

    tensors = {
        "xyz": xyz.detach().cpu().contiguous().float(),
        "features_dc": features_dc.detach().cpu().contiguous().float(),
        "scaling": scaling.detach().cpu().contiguous().float(),
        "rotation": rotation.detach().cpu().contiguous().float(),
        "opacity": opacity.detach().cpu().contiguous().float(),
        "semantic_weights": semantic_weights.contiguous(),
        "semantic_indices": semantic_indices.contiguous(),
        "semantic_codebooks": semantic_codebooks,
    }
    if embedded_source_ids is not None:
        tensors["source_ids"] = embedded_source_ids
    tensors.update(sh_tensors)
    bundle = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "iteration": int(iteration),
        "active_sh_degree": int(active_sh_degree),
        "max_sh_degree": 3,
        "point_count": point_count,
        "topk": int(topk),
        "semantic_level_count": int(
            1 if semantic_logits.ndim == 2 else semantic_logits.shape[1]
        ),
        "sh_mode": sh_mode,
        "sh_feature_shape": stored_feature_shape,
        "tensors": tensors,
    }
    if source_id_lineage is not None:
        bundle["source_id_lineage"] = source_id_lineage
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_path)

    logical_bytes = {name: _tensor_nbytes(value) for name, value in tensors.items()}
    manifest = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "scene_artifact": os.path.basename(output_path),
        "scene_bytes": int(os.path.getsize(output_path)),
        "shared_bytes": 0,
        "one_scene_total_bytes": int(os.path.getsize(output_path)),
        "sha256": _sha256(output_path),
        "point_count": point_count,
        "topk": int(topk),
        "logical_tensor_bytes": logical_bytes,
        "logical_tensor_bytes_total": int(sum(logical_bytes.values())),
        "tensor_schema": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in tensors.items()
        },
        "excluded_training_state": [
            "optimizer", "dense_semantic_logits", "gradient_accumulators"
        ] + (["unquantized_features_rest"] if sh_quantization_path else []),
        "source": {
            "checkpoint": os.path.abspath(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "sh_quantization": (
                os.path.abspath(sh_quantization_path)
                if sh_quantization_path else None
            ),
            "iteration": int(iteration),
        },
    }
    if source_id_lineage is not None:
        manifest["source_id_lineage"] = {
            **source_id_lineage,
            "source_sidecar_provenance": {
                "path": os.path.abspath(source_id_sidecar_path),
                "bytes": int(os.path.getsize(source_id_sidecar_path)),
                "sha256": _sha256(source_id_sidecar_path),
                "needed_for_deployment": False,
            },
        }
    with open(manifest_path, "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
    return manifest


def load_compact_gaussians(artifact_path, device="cuda"):
    """Load a compact scene without consulting a training checkpoint."""
    from torch import nn
    from scene.gaussian_model import GaussianModel

    bundle = torch.load(artifact_path, map_location="cpu")
    if bundle.get("format") != FORMAT_NAME or bundle.get("version") != FORMAT_VERSION:
        raise ValueError("unsupported compact artifact format")
    tensors = bundle["tensors"]
    point_count = int(bundle["point_count"])
    if tensors["xyz"].shape[0] != point_count:
        raise ValueError("artifact point count is inconsistent")

    model = GaussianModel(int(bundle["max_sh_degree"]))
    model.active_sh_degree = int(bundle["active_sh_degree"])
    model._xyz = nn.Parameter(tensors["xyz"].to(device), requires_grad=False)
    model._features_dc = nn.Parameter(
        tensors["features_dc"].to(device), requires_grad=False
    )
    if bundle.get("sh_mode", "vq_uint8") == "full_float32":
        sh_values = tensors["features_rest"]
    else:
        sh_indices = tensors["sh_indices"].to(torch.int64)
        sh_values = tensors["sh_codebook"][sh_indices]
        sh_values = sh_values.reshape(point_count, *bundle["sh_feature_shape"])
    model._features_rest = nn.Parameter(sh_values.to(device), requires_grad=False)
    model._scaling = nn.Parameter(tensors["scaling"].to(device), requires_grad=False)
    model._rotation = nn.Parameter(tensors["rotation"].to(device), requires_grad=False)
    model._opacity = nn.Parameter(tensors["opacity"].to(device), requires_grad=False)
    model._language_feature_codebooks = nn.Parameter(
        tensors["semantic_codebooks"].float().to(device), requires_grad=False
    )
    model._language_feature_weights = tensors["semantic_weights"].float().to(device)
    model._language_feature_indices = tensors["semantic_indices"].float().to(device)
    return model, bundle


def validate_artifact(
    artifact_path, checkpoint_path=None, source_id_sidecar_path=None
):
    bundle = torch.load(artifact_path, map_location="cpu")
    if bundle.get("format") != FORMAT_NAME or bundle.get("version") != FORMAT_VERSION:
        raise ValueError("unsupported compact artifact format")
    tensors = bundle["tensors"]
    point_count = int(bundle["point_count"])
    point_tensors = (
        "xyz", "features_dc", "scaling", "rotation", "opacity",
        "semantic_weights", "semantic_indices"
    )
    if bundle.get("sh_mode", "vq_uint8") == "full_float32":
        point_tensors += ("features_rest",)
    else:
        point_tensors += ("sh_indices",)
    for name in point_tensors:
        if tensors[name].shape[0] != point_count:
            raise ValueError("{} has an inconsistent point count".format(name))
    semantic_levels = int(bundle["semantic_level_count"])
    expected_sparse_width = semantic_levels * int(bundle["topk"])
    if tensors["semantic_weights"].shape[1] != expected_sparse_width:
        raise ValueError("semantic weight width is inconsistent")
    if tensors["semantic_indices"].shape != tensors["semantic_weights"].shape:
        raise ValueError("semantic indices and weights have different shapes")
    if not torch.isfinite(tensors["semantic_weights"].float()).all():
        raise ValueError("semantic weights contain non-finite values")
    lineage = bundle.get("source_id_lineage")
    lineage_audit = None
    if lineage is not None:
        if "source_ids" not in tensors:
            raise ValueError("lineage metadata is present but source_ids is not embedded")
        source_ids = tensors["source_ids"]
        if source_ids.dtype != torch.int64 or source_ids.ndim != 1:
            raise ValueError("embedded source_ids must be one-dimensional int64")
        if source_ids.shape[0] != point_count:
            raise ValueError("embedded source_ids and compact point count disagree")
        origin_count = int(lineage["origin_point_count"])
        if source_ids.numel():
            if int(source_ids.min()) < 0 or int(source_ids.max()) >= origin_count:
                raise ValueError("embedded source_ids are outside the origin range")
            if source_ids.numel() > 1 and not bool(
                torch.all(source_ids[1:] > source_ids[:-1])
            ):
                raise ValueError("embedded source_ids are not strictly increasing")
        source_ids_sha256 = _sha256_tensor(source_ids)
        if source_ids_sha256 != lineage["remaining_source_ids_sha256"]:
            raise ValueError("embedded source_ids hash differs from lineage metadata")
        xyz_sha256 = _sha256_tensor(tensors["xyz"])
        if xyz_sha256 != lineage["bound_xyz_sha256"]:
            raise ValueError("compact XYZ row order differs from lineage binding")
        if int(lineage["remaining_point_count"]) != point_count:
            raise ValueError("lineage remaining point count is inconsistent")
        if origin_count - point_count != int(lineage["deleted_source_point_count"]):
            raise ValueError("lineage deleted point count is inconsistent")
        if not lineage.get("embedded_in_deployment_artifact"):
            raise ValueError("lineage is not declared embedded in deployment artifact")
        if not lineage.get("counted_in_deployment_bytes"):
            raise ValueError("embedded source IDs are not counted as deployment bytes")
        if int(lineage["embedded_logical_bytes"]) != _tensor_nbytes(source_ids):
            raise ValueError("embedded source-ID byte count is inconsistent")

        external_exact = None
        checkpoint_exact = None
        if checkpoint_path is not None:
            if _sha256(checkpoint_path) != lineage["bound_checkpoint_sha256"]:
                raise ValueError("checkpoint differs from embedded lineage binding")
            checkpoint_params, _ = torch.load(checkpoint_path, map_location="cpu")
            if _sha256_tensor(checkpoint_params[1]) != xyz_sha256:
                raise ValueError("checkpoint XYZ differs from compact source-ID rows")
            checkpoint_exact = True
        if source_id_sidecar_path is not None:
            from source_id_lineage import load_lineage_sidecar

            external_ids, external_bundle = load_lineage_sidecar(
                source_id_sidecar_path, checkpoint_path=checkpoint_path
            )
            if not torch.equal(external_ids, source_ids):
                raise ValueError("embedded and external source IDs are not exact")
            if (
                external_bundle["origin_checkpoint_sha256"]
                != lineage["origin_checkpoint_sha256"]
                or external_bundle["origin_xyz_sha256"]
                != lineage["origin_xyz_sha256"]
            ):
                raise ValueError("embedded and external lineage origins differ")
            external_exact = True
        lineage_audit = {
            "embedded": True,
            "tensor": "source_ids",
            "dtype": str(source_ids.dtype),
            "logical_bytes": _tensor_nbytes(source_ids),
            "source_ids_sha256": source_ids_sha256,
            "xyz_row_sha256": xyz_sha256,
            "strictly_increasing": True,
            "checkpoint_binding_verified": checkpoint_exact,
            "external_sidecar_exact": external_exact,
            "counted_in_serialized_artifact_bytes": True,
        }
    elif checkpoint_path is not None or source_id_sidecar_path is not None:
        raise ValueError("requested lineage audit but compact artifact has no lineage")

    return {
        "scene_bytes": int(os.path.getsize(artifact_path)),
        "sha256": _sha256(artifact_path),
        "point_count": point_count,
        "topk": int(bundle["topk"]),
        "source_id_lineage": lineage_audit,
    }


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--sh-quantization", default=None)
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--topk", type=int, default=4)
    export_parser.add_argument("--source-id-sidecar", default=None)
    export_parser.add_argument("--force", action="store_true")
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--artifact", required=True)
    validate_parser.add_argument("--checkpoint", default=None)
    validate_parser.add_argument("--source-id-sidecar", default=None)
    args = parser.parse_args()

    if args.command == "export":
        result = export_compact_artifact(
            args.checkpoint, args.sh_quantization, args.output,
            topk=args.topk, force=args.force,
            source_id_sidecar_path=args.source_id_sidecar,
        )
    else:
        result = validate_artifact(
            args.artifact,
            checkpoint_path=args.checkpoint,
            source_id_sidecar_path=args.source_id_sidecar,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
