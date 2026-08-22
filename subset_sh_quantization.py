#!/usr/bin/env python3
"""Subset a materialized VQ-first SH code by tracked host-row IDs."""

import argparse
import os
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-quantization", required=True)
    parser.add_argument("--lineage-sidecar", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = torch.load(args.source_quantization, map_location="cpu")
    lineage = torch.load(args.lineage_sidecar, map_location="cpu")
    source_ids = lineage["remaining_source_ids"].to(torch.int64)
    source_indices = source["indices"]
    if source_ids.numel() and int(source_ids.max()) >= source_indices.shape[0]:
        raise ValueError("lineage IDs exceed the VQ-first assignment rows")
    result = dict(source)
    result["indices"] = source_indices[source_ids].contiguous()
    temporary = args.output + ".tmp"
    torch.save(result, temporary)
    os.replace(temporary, args.output)
    print(
        "wrote {} assignments from {} source rows".format(
            result["indices"].shape[0], source_indices.shape[0]
        )
    )


if __name__ == "__main__":
    main()
