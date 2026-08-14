"""Merge three released single-scale LangSplatV2 checkpoints into one model."""

import argparse
import os

import torch


GEOMETRY_FIELDS = (1, 2, 3, 4, 5, 6, 9, 10, 11)


def load_checkpoint(path):
    payload = torch.load(path)
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise ValueError("Unexpected checkpoint payload in {}".format(path))
    model_params, iteration = payload
    if len(model_params) != 14:
        raise ValueError(
            "Expected a 14-field semantic checkpoint, got {} in {}".format(
                len(model_params), path
            )
        )
    return tuple(model_params), iteration


def validate_single_scale(model_params, path):
    logits = model_params[7]
    codebooks = model_params[8]
    if logits.ndim != 2 or codebooks.ndim != 3:
        raise ValueError(
            "{} is not a released single-scale checkpoint: logits {}, codebooks {}".format(
                path, tuple(logits.shape), tuple(codebooks.shape)
            )
        )
    if logits.shape[1] != codebooks.shape[0] * codebooks.shape[1]:
        raise ValueError("Logit/codebook dimensions do not match in {}".format(path))


def tensors_identical(left, right):
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(left, right)
    )


def merge_checkpoints(paths, output_path, check_geometry=True):
    reference, reference_iteration = load_checkpoint(paths[0])
    validate_single_scale(reference, paths[0])
    logits = [reference[7]]
    codebooks = [reference[8]]
    for scale_index, path in enumerate(paths[1:], start=1):
        model, iteration = load_checkpoint(path)
        validate_single_scale(model, path)
        if iteration != reference_iteration:
            raise ValueError(
                "Checkpoint iterations differ: {} and {}".format(
                    reference_iteration, iteration
                )
            )
        if check_geometry:
            for field in GEOMETRY_FIELDS:
                if not tensors_identical(reference[field], model[field]):
                    raise ValueError(
                        "Geometry field {} differs between {} and {}".format(
                            field, paths[0], paths[scale_index]
                        )
                    )
        logits.append(model[7])
        codebooks.append(model[8])

    joint_logits = torch.stack(logits, dim=1)
    joint_codebooks = torch.stack(codebooks, dim=0)
    merged = list(reference)
    merged[7] = joint_logits
    merged[8] = joint_codebooks
    # The three source optimizers have unrelated semantic parameters.  The
    # joint compression stage intentionally creates a fresh optimizer.
    merged[12] = {"state": {}, "param_groups": []}

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    torch.save((tuple(merged), reference_iteration), output_path)
    print("Merged semantic logits: {}".format(tuple(joint_logits.shape)))
    print("Merged semantic codebooks: {}".format(tuple(joint_codebooks.shape)))
    print("Saved joint checkpoint to {}".format(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge small/medium/large LangSplatV2 checkpoints"
    )
    parser.add_argument("checkpoints", nargs=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip_geometry_check", action="store_true")
    arguments = parser.parse_args()
    merge_checkpoints(
        arguments.checkpoints,
        arguments.output,
        check_geometry=not arguments.skip_geometry_check,
    )
