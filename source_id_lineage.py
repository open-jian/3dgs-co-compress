"""Evaluation-only source-row lineage for controlled ClunoGS runs.

The compact model may prune Gaussians.  A nearest-neighbour reconstruction of
the deleted rows would hide those deletions, so controlled direct-3D runs keep
an integer map from each remaining row back to its RGB-host source row.  The
sidecar is not a deployment payload and is never added to model size.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import argparse
from pathlib import Path

import torch


FORMAT_NAME = "clunogs.source-id-lineage"
FORMAT_VERSION = 1
GEOMETRY_FIELDS = (
    "xyz",
    "features_dc",
    "features_rest",
    "scaling",
    "rotation",
    "opacity",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    if tensor.ndim == 0:
        digest.update(tensor.numpy().tobytes())
    else:
        row_bytes = max(1, tensor[0].numel() * tensor.element_size())
        rows_per_chunk = max(1, (8 * 1024 * 1024) // row_bytes)
        for start in range(0, tensor.shape[0], rows_per_chunk):
            digest.update(tensor[start : start + rows_per_chunk].numpy().tobytes())
    return digest.hexdigest()


def load_checkpoint(path, map_location="cpu"):
    kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    payload = torch.load(path, **kwargs)
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise ValueError("checkpoint must contain (model_params, iteration)")
    model_params, iteration = payload
    if len(model_params) not in (12, 14):
        raise ValueError("source-ID tracking expects a 12- or 14-field checkpoint")
    return model_params, int(iteration)


def geometry_tensors(model_params):
    return dict(zip(GEOMETRY_FIELDS, model_params[1:7]))


def compare_geometry(left_params, right_params):
    left = geometry_tensors(left_params)
    right = geometry_tensors(right_params)
    matches = {}
    for name in GEOMETRY_FIELDS:
        lhs = left[name].detach().cpu()
        rhs = right[name].detach().cpu()
        matches[name] = bool(lhs.shape == rhs.shape and torch.equal(lhs, rhs))
    return matches


def _validate_ids(source_ids, origin_point_count, expected_rows):
    source_ids = source_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
    if source_ids.ndim != 1 or source_ids.shape[0] != expected_rows:
        raise ValueError(
            "source IDs must be one-dimensional and match the current Gaussian count"
        )
    if source_ids.numel():
        if int(source_ids.min()) < 0 or int(source_ids.max()) >= origin_point_count:
            raise ValueError("source IDs are outside the origin checkpoint range")
        if source_ids.numel() > 1 and not bool(torch.all(source_ids[1:] > source_ids[:-1])):
            raise ValueError(
                "source IDs must remain strictly increasing; reordering/duplication occurred"
            )
    return source_ids


def initialize_lineage(
    gaussians,
    start_checkpoint,
    origin_checkpoint=None,
    input_sidecar=None,
):
    """Attach validated IDs to a restored GaussianModel and return provenance."""
    start_checkpoint = os.path.abspath(start_checkpoint)
    start_params, start_iteration = load_checkpoint(start_checkpoint)
    current_count = int(gaussians.get_xyz.shape[0])
    if int(start_params[1].shape[0]) != current_count:
        raise ValueError("restored model count differs from the start checkpoint")

    if input_sidecar:
        input_sidecar = os.path.abspath(input_sidecar)
        bundle = torch.load(input_sidecar, map_location="cpu")
        if bundle.get("format") != FORMAT_NAME or bundle.get("version") != FORMAT_VERSION:
            raise ValueError("unsupported source-ID lineage sidecar")
        if bundle["bound_checkpoint_sha256"] != sha256_file(start_checkpoint):
            raise ValueError("input source-ID sidecar is not bound to --start_checkpoint")
        if bundle["bound_xyz_sha256"] != sha256_tensor(start_params[1]):
            raise ValueError("input source-ID sidecar has the wrong checkpoint row order")
        origin_count = int(bundle["origin_point_count"])
        source_ids = _validate_ids(
            bundle["remaining_source_ids"], origin_count, current_count
        )
        metadata = {
            "origin_checkpoint": bundle["origin_checkpoint"],
            "origin_checkpoint_sha256": bundle["origin_checkpoint_sha256"],
            "origin_xyz_sha256": bundle["origin_xyz_sha256"],
            "origin_point_count": origin_count,
            "resumed_from_sidecar": input_sidecar,
            "resumed_from_sidecar_sha256": sha256_file(input_sidecar),
        }
    else:
        # With no incoming sidecar, the explicitly selected start checkpoint
        # defines the host row order for this controlled run.  This is also
        # valid for an unpruned semantic host: IDs are local to that host and
        # do not claim ancestry before it.  Resuming a previously tracked run
        # still requires input_sidecar so its earlier origin is preserved.
        origin_checkpoint = os.path.abspath(origin_checkpoint or start_checkpoint)
        origin_params, _ = load_checkpoint(origin_checkpoint)
        geometry_exact = compare_geometry(start_params, origin_params)
        if not all(geometry_exact.values()):
            mismatched = [name for name, value in geometry_exact.items() if not value]
            raise ValueError(
                "RGB host differs from source-ID origin geometry: {}".format(
                    ", ".join(mismatched)
                )
            )
        origin_count = int(origin_params[1].shape[0])
        if origin_count != current_count:
            raise ValueError("fresh source-ID tracking requires one row per origin row")
        source_ids = torch.arange(origin_count, dtype=torch.int64)
        metadata = {
            "origin_checkpoint": origin_checkpoint,
            "origin_checkpoint_sha256": sha256_file(origin_checkpoint),
            "origin_xyz_sha256": sha256_tensor(origin_params[1]),
            "origin_point_count": origin_count,
            "origin_geometry_exact_match": geometry_exact,
            "resumed_from_sidecar": None,
            "resumed_from_sidecar_sha256": None,
        }

    gaussians.enable_source_id_tracking(source_ids, metadata["origin_point_count"])
    metadata.update(
        {
            "start_checkpoint": start_checkpoint,
            "start_checkpoint_sha256": sha256_file(start_checkpoint),
            "start_iteration": start_iteration,
            "start_point_count": current_count,
        }
    )
    return metadata


def save_lineage_sidecar(gaussians, checkpoint_path, metadata, output_path=None):
    checkpoint_path = os.path.abspath(checkpoint_path)
    if output_path is None:
        output_path = checkpoint_path + ".source_ids.pt"
    output_path = os.path.abspath(output_path)
    manifest_path = output_path + ".manifest.json"
    if os.path.exists(output_path) or os.path.exists(manifest_path):
        raise FileExistsError("refusing to overwrite source-ID sidecar")
    if gaussians._source_ids is None:
        raise ValueError("source-ID tracking is not enabled")

    checkpoint_params, iteration = load_checkpoint(checkpoint_path)
    point_count = int(checkpoint_params[1].shape[0])
    source_ids = _validate_ids(
        gaussians._source_ids,
        int(metadata["origin_point_count"]),
        point_count,
    )
    current_xyz_sha256 = sha256_tensor(gaussians.get_xyz)
    checkpoint_xyz_sha256 = sha256_tensor(checkpoint_params[1])
    if current_xyz_sha256 != checkpoint_xyz_sha256:
        raise ValueError("checkpoint XYZ differs from the in-memory tracked row order")

    bundle = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "evaluation_only": True,
        "counted_in_deployment_bytes": False,
        "mapping": "final/checkpoint row index -> original RGB-host source row index",
        "origin_checkpoint": metadata["origin_checkpoint"],
        "origin_checkpoint_sha256": metadata["origin_checkpoint_sha256"],
        "origin_xyz_sha256": metadata["origin_xyz_sha256"],
        "origin_point_count": int(metadata["origin_point_count"]),
        "bound_checkpoint": checkpoint_path,
        "bound_checkpoint_sha256": sha256_file(checkpoint_path),
        "bound_xyz_sha256": checkpoint_xyz_sha256,
        "bound_iteration": iteration,
        "remaining_point_count": point_count,
        "deleted_source_point_count": int(metadata["origin_point_count"]) - point_count,
        "remaining_source_ids": source_ids,
        "lineage_start": metadata,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path + ".tmp"
    torch.save(bundle, temporary)
    os.replace(temporary, output_path)
    manifest = {
        key: value
        for key, value in bundle.items()
        if key not in ("remaining_source_ids", "lineage_start")
    }
    manifest.update(
        {
            "source_id_sidecar": output_path,
            "source_id_sidecar_bytes": os.path.getsize(output_path),
            "source_id_sidecar_sha256": sha256_file(output_path),
            "remaining_source_ids_sha256": sha256_tensor(source_ids),
            "remaining_source_id_min": int(source_ids.min()) if source_ids.numel() else None,
            "remaining_source_id_max": int(source_ids.max()) if source_ids.numel() else None,
            "remaining_source_ids_strictly_increasing": True,
            "lineage_start": metadata,
        }
    )
    temporary_manifest = manifest_path + ".tmp"
    with open(temporary_manifest, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_manifest, manifest_path)
    print(
        "Saved source-ID sidecar: {} -> {} rows ({} deleted)".format(
            metadata["origin_point_count"], point_count,
            int(metadata["origin_point_count"]) - point_count,
        )
    )
    return output_path, manifest


def load_lineage_sidecar(sidecar_path, checkpoint_path=None):
    sidecar_path = os.path.abspath(sidecar_path)
    bundle = torch.load(sidecar_path, map_location="cpu")
    if bundle.get("format") != FORMAT_NAME or bundle.get("version") != FORMAT_VERSION:
        raise ValueError("unsupported source-ID lineage sidecar")
    source_ids = _validate_ids(
        bundle["remaining_source_ids"],
        int(bundle["origin_point_count"]),
        int(bundle["remaining_point_count"]),
    )
    if checkpoint_path is not None:
        checkpoint_path = os.path.abspath(checkpoint_path)
        if bundle["bound_checkpoint_sha256"] != sha256_file(checkpoint_path):
            raise ValueError("source-ID sidecar is not bound to the requested checkpoint")
        checkpoint_params, _ = load_checkpoint(checkpoint_path)
        if bundle["bound_xyz_sha256"] != sha256_tensor(checkpoint_params[1]):
            raise ValueError("source-ID sidecar checkpoint row order mismatch")
    return source_ids, bundle


def validate_lineage_sidecar(sidecar_path, checkpoint_path, compact_artifact=None):
    source_ids, bundle = load_lineage_sidecar(sidecar_path, checkpoint_path)
    origin_count = int(bundle["origin_point_count"])
    deleted = torch.ones(origin_count, dtype=torch.bool)
    deleted[source_ids] = False
    result = {
        "schema": "clunogs_source_id_lineage_audit_v1",
        "pass": True,
        "sidecar": os.path.abspath(sidecar_path),
        "sidecar_bytes": os.path.getsize(sidecar_path),
        "sidecar_sha256": sha256_file(sidecar_path),
        "bound_checkpoint": os.path.abspath(checkpoint_path),
        "bound_checkpoint_sha256": sha256_file(checkpoint_path),
        "origin_checkpoint": bundle["origin_checkpoint"],
        "origin_checkpoint_sha256": bundle["origin_checkpoint_sha256"],
        "origin_point_count": origin_count,
        "remaining_point_count": int(source_ids.shape[0]),
        "deleted_source_point_count": int(deleted.sum()),
        "remaining_source_ids_sha256": sha256_tensor(source_ids),
        "deleted_source_mask_sha256": sha256_tensor(deleted),
        "source_ids_strictly_increasing": True,
        "mapping": bundle["mapping"],
        "evaluation_only": True,
        "counted_in_deployment_bytes": False,
    }
    if compact_artifact:
        compact_artifact = os.path.abspath(compact_artifact)
        compact = torch.load(compact_artifact, map_location="cpu")
        if compact.get("format") != "clunogs.compact-scene":
            raise ValueError("unsupported ClunoGS compact artifact")
        if int(compact["point_count"]) != int(source_ids.shape[0]):
            raise ValueError("compact artifact and source-ID sidecar row counts differ")
        compact_xyz_sha256 = sha256_tensor(compact["tensors"]["xyz"])
        if compact_xyz_sha256 != bundle["bound_xyz_sha256"]:
            raise ValueError("compact artifact reordered or changed checkpoint XYZ")
        if "source_ids" not in compact["tensors"]:
            raise ValueError("compact artifact does not embed source_ids")
        compact_source_ids = compact["tensors"]["source_ids"]
        if compact_source_ids.dtype != torch.int64:
            raise ValueError("compact embedded source_ids are not int64")
        if not torch.equal(compact_source_ids.cpu(), source_ids):
            raise ValueError("compact embedded and external source IDs are not exact")
        compact_lineage = compact.get("source_id_lineage")
        if not compact_lineage:
            raise ValueError("compact artifact lacks embedded lineage metadata")
        if not compact_lineage.get("embedded_in_deployment_artifact"):
            raise ValueError("compact lineage is not marked as embedded")
        if not compact_lineage.get("counted_in_deployment_bytes"):
            raise ValueError("compact source IDs are not counted as deployment bytes")
        if (
            compact_lineage.get("remaining_source_ids_sha256")
            != sha256_tensor(source_ids)
        ):
            raise ValueError("compact source-ID metadata hash mismatch")
        result["compact_artifact"] = {
            "path": compact_artifact,
            "bytes": os.path.getsize(compact_artifact),
            "sha256": sha256_file(compact_artifact),
            "point_count": int(compact["point_count"]),
            "xyz_sha256": compact_xyz_sha256,
            "row_order_matches_bound_checkpoint": True,
            "embedded_source_ids_exact": True,
            "source_ids_counted_in_serialized_artifact_bytes": True,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--compact-artifact", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = validate_lineage_sidecar(
        args.sidecar, args.checkpoint, compact_artifact=args.compact_artifact
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if os.path.exists(args.output):
            raise FileExistsError("refusing to overwrite lineage audit output")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as handle:
            handle.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
