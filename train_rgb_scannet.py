"""Short-run RGB-only 3DGS training entry point for native ScanNet scenes."""

import sys
from argparse import ArgumentParser

import torch

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui
from train import training
from utils.general_utils import safe_state


def main():
    parser = ArgumentParser(description="RGB-only ScanNet 3DGS training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=55559)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--accum_iter", type=int, default=1)
    parser.add_argument("--topk", type=int, default=1)
    args = parser.parse_args(sys.argv[1:])

    # The released parser exposes include_feature only as a store_true flag
    # whose default is True. This dedicated entry point makes the RGB-only
    # protocol explicit without changing the semantic training CLI.
    args.include_feature = False
    args.save_iterations = sorted(set(args.save_iterations + [args.iterations]))
    args.checkpoint_iterations = sorted(
        set(args.checkpoint_iterations + [args.iterations])
    )
    safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        None,
        args.debug_from,
        args,
    )


if __name__ == "__main__":
    main()
