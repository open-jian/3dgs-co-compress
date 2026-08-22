#!/usr/bin/env python3
"""Subset a materialized VQ-first sidecar by tracked host-row IDs."""

import argparse
import os
import torch

from c3dgs_quantization import QUANTIZATION_FORMAT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-quantization", required=True)
    parser.add_argument("--lineage-sidecar", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = torch.load(args.source_quantization, map_location="cpu")
    lineage = torch.load(args.lineage_sidecar, map_location="cpu")
    source_ids = lineage["remaining_source_ids"].to(torch.int64)
    if source.get("format") == QUANTIZATION_FORMAT:
        source_point_count = int(source["point_count"])
        if source_ids.numel() and int(source_ids.max()) >= source_point_count:
            raise ValueError("lineage IDs exceed the VQ-first sidecar rows")
        result = dict(source)
        result["blocks"] = dict(source["blocks"])
        for block_name in ("color", "covariance"):
            block = dict(source["blocks"][block_name])
            block["indices"] = block["indices"][source_ids].contiguous()
            result["blocks"][block_name] = block
        semantic_blocks = []
        for source_block in source["blocks"].get("semantic", []):
            block = dict(source_block)
            block["indices"] = block["indices"][source_ids].contiguous()
            semantic_blocks.append(block)
        if semantic_blocks:
            result["blocks"]["semantic"] = semantic_blocks
        result["scale_factor"] = source["scale_factor"][source_ids].contiguous()
        if source.get("semantic_fixed_indices") is not None:
            result["semantic_fixed_indices"] = source[
                "semantic_fixed_indices"
            ][source_ids].contiguous()
        result["point_count"] = int(source_ids.numel())
        original_rows = source_point_count
    else:
        source_indices = source["indices"]
        if source_ids.numel() and int(source_ids.max()) >= source_indices.shape[0]:
            raise ValueError("lineage IDs exceed the VQ-first assignment rows")
        result = dict(source)
        result["indices"] = source_indices[source_ids].contiguous()
        original_rows = source_indices.shape[0]
    temporary = args.output + ".tmp"
    torch.save(result, temporary)
    os.replace(temporary, args.output)
    print(
        "wrote {} assignments from {} source rows".format(
            source_ids.shape[0], original_rows
        )
    )


if __name__ == "__main__":
    main()
