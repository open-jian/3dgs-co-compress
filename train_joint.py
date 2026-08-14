"""Joint three-scale LangSplatV2 + CoLaSplat optimization.

This entry point keeps one Gaussian geometry and three semantic VQ heads.  The
three heads are supervised together, then every pruning event applies one mask
to geometry, RGB attributes, opacity, and all semantic logits.
"""

import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import torch
from tqdm import tqdm

from admm import ADMM, get_pruning_mask
from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.loss_utils import cos_loss, l1_loss, ssim
from utils.vq_utils import (
    ResidualVectorQuantizationWithClustering,
    load_2d_language_feature,
)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def workspace_output_root():
    return os.path.abspath(
        os.environ.get("OUTPUT_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "Output"))
    )


def prepare_output_and_logger(args):
    if not args.model_path:
        unique = os.getenv("OAR_JOB_ID", str(uuid.uuid4()))
        args.model_path = os.path.join(workspace_output_root(), "langsplatv2_cluno", "joint", unique[:10])
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_file:
        cfg_file.write(str(Namespace(**vars(args))))
    return SummaryWriter(args.model_path) if TENSORBOARD_FOUND else None


def initialize_scale_codebooks(gaussians, dataset, opt, feature_levels):
    device = torch.device("cuda")
    codebooks = gaussians._semantic_codebooks()
    for semantic_level, feature_level in enumerate(feature_levels):
        print("Initializing semantic level {} from 2D level {}".format(
            semantic_level, feature_level
        ))
        features = load_2d_language_feature(
            dataset.lf_path, device, feature_level=feature_level
        )
        rvq = ResidualVectorQuantizationWithClustering(
            opt.vq_layer_num, opt.codebook_size, features.shape[1], device
        ).to(device)
        rvq.fit_quantizers(features.float())
        initialized = torch.stack(rvq.quantizers, dim=0).to(device)
        with torch.no_grad():
            codebooks[semantic_level].copy_(initialized)
        del features, rvq, initialized


def semantic_reconstruction_loss(prediction, target, mask, mode):
    loss = prediction.new_zeros(())
    if mode in ("cos", "cos+l1"):
        loss = loss + cos_loss(prediction * mask, target * mask)
    if mode in ("l1", "cos+l1"):
        loss = loss + l1_loss(prediction * mask, target * mask)
    return loss


def save_sh_quantization(admm, model_path, iteration):
    if admm is None or admm.sh_projector.centers is None:
        return
    # Reassign the final (possibly fine-tuned after the last ADMM update) SH
    # vectors to the frozen codebook before serializing the compact indices.
    admm.sh_projector.project(
        admm.gaussian_model._features_rest, update_centers=False
    )
    output_path = os.path.join(model_path, "sh_quantization_{}.pth".format(iteration))
    torch.save(admm.sh_projector.state_dict(), output_path)
    print("Saved SH codebook and indices to {}".format(output_path))


def training(
    dataset,
    opt,
    pipe,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    args,
):
    feature_levels = tuple(args.feature_levels)
    if len(feature_levels) != 3:
        raise ValueError("Joint training requires exactly three feature levels")
    if len(set(feature_levels)) != len(feature_levels):
        raise ValueError("feature_levels must be distinct")
    if opt.vq_layer_num * opt.codebook_size != 64:
        raise ValueError(
            "The released training rasterizer has 64 semantic channels; "
            "vq_layer_num * codebook_size must equal 64"
        )
    if args.topk <= 0 or args.topk > opt.codebook_size:
        raise ValueError("topk must be in [1, codebook_size]")

    opt.semantic_level_num = len(feature_levels)
    opt.joint_optimize = not args.semantic_only
    if args.semantic_only:
        args.no_rgb_loss = True
        args.no_admm_loss = True
    args.semantic_level_num = opt.semantic_level_num
    args.joint_optimize = opt.joint_optimize

    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    first_iter = 0
    checkpoint_has_semantics = False
    if checkpoint:
        model_params, first_iter = torch.load(checkpoint)
        checkpoint_has_semantics = len(model_params) == 14
        if len(model_params) == 12:
            first_iter = 0
        elif len(model_params) != 14:
            raise ValueError("Unsupported checkpoint tuple length: {}".format(len(model_params)))
        if checkpoint_has_semantics:
            checkpoint_logits = model_params[7]
            checkpoint_semantic_levels = (
                1 if checkpoint_logits.ndim == 2 else checkpoint_logits.shape[1]
            )
            if checkpoint_semantic_levels != 3:
                raise ValueError(
                    "The semantic checkpoint contains {} scale(s), expected 3. "
                    "Start from the RGB checkpoint or a joint checkpoint.".format(
                        checkpoint_semantic_levels
                    )
                )
        gaussians.restore(model_params, opt)
        if args.reset_iteration:
            first_iter = 0
    elif opt.include_feature:
        raise ValueError("--start_checkpoint must point to an RGB or joint checkpoint")

    if opt.include_feature and not checkpoint_has_semantics:
        initialize_scale_codebooks(gaussians, dataset, opt, feature_levels)

    background_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(background_color, dtype=torch.float32, device="cuda")
    viewpoint_stack = None
    ema_total = 0.0
    ema_rgb = 0.0
    ema_semantic = [0.0] * len(feature_levels)
    admm = None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Joint training")

    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifier = network_gui.receive()
                image_bytes = None
                if custom_cam is not None:
                    gui_image = render(
                        custom_cam, gaussians, pipe, background, opt, scaling_modifier
                    )["render"]
                    image_bytes = memoryview(
                        (torch.clamp(gui_image, 0, 1) * 255)
                        .byte().permute(1, 2, 0).contiguous().cpu().numpy()
                    )
                network_gui.send(image_bytes, dataset.source_path)
                if do_training and (iteration < opt.iterations or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        if (
            not args.semantic_only
            and opt.enable_admm_loss
            and opt.admm_start_iter <= iteration <= opt.admm_end_iter
        ):
            if admm is None:
                print("[ITER {}] Initializing ADMM".format(iteration))
                admm = ADMM(
                    gaussians,
                    opt.rho_opacity,
                    opt.rho_sh,
                    opt.sh_codebook_size,
                )
                admm.update_opacity(opt.pruning_fraction2, update_dual=False)
                admm.update_sh(update_dual=False, update_centers=True)
            elif iteration % opt.admm_interval == 0:
                admm.update_opacity(opt.pruning_fraction2, update_dual=True)
                admm.update_sh(
                    update_dual=True,
                    update_centers=iteration < opt.freeze_sh_codebook_iter,
                )

        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if iteration - 1 == debug_from:
            pipe.debug = True

        targets = viewpoint.get_language_features(dataset.lf_path, feature_levels)
        semantic_losses = []
        rgb_image = None
        for semantic_level, (target, mask) in enumerate(targets):
            render_package = render(
                viewpoint,
                gaussians,
                pipe,
                background,
                opt,
                semantic_level=semantic_level,
            )
            if rgb_image is None:
                rgb_image = render_package["render"]
            weight_map = render_package["language_feature_weight_map"]
            rvq_layers = gaussians._semantic_codebooks().shape[1]
            rvq_layer = min(iteration * rvq_layers // opt.iterations, rvq_layers - 1)
            prediction = gaussians.compute_layer_feature_map(
                weight_map, rvq_layer, semantic_level=semantic_level
            )
            if args.normalize:
                prediction = prediction / (prediction.norm(dim=0, keepdim=True) + 1e-10)
            semantic_losses.append(
                semantic_reconstruction_loss(
                    prediction, target, mask, args.semantic_loss
                )
            )

        language_loss = torch.stack(semantic_losses).sum()
        target_rgb = viewpoint.original_image.cuda()
        rgb_l1 = l1_loss(rgb_image, target_rgb)
        rgb_loss = (
            (1.0 - opt.lambda_dssim) * rgb_l1
            + opt.lambda_dssim * (1.0 - ssim(rgb_image, target_rgb))
        )
        loss = rgb_image.new_zeros(())
        if opt.enable_language_loss and not args.no_language_loss:
            loss = loss + opt.language_loss_coeff * language_loss
        if opt.enable_rgb_loss and not args.no_rgb_loss:
            loss = loss + opt.rgb_loss_coeff * rgb_loss

        opacity_admm_loss = rgb_image.new_zeros(())
        sh_admm_loss = rgb_image.new_zeros(())
        if (
            admm is not None
            and opt.enable_admm_loss
            and not args.no_admm_loss
            and iteration % opt.admm_interval == 0
            and iteration <= opt.admm_end_iter
        ):
            opacity_admm_loss = admm.opacity_loss()
            sh_admm_loss = admm.sh_loss()
            loss = loss + opt.admm_loss_coeff * (
                opacity_admm_loss + sh_admm_loss
            )
        if not loss.requires_grad:
            raise RuntimeError("All training losses are disabled")
        loss.backward()

        with torch.no_grad():
            if (
                not args.semantic_only
                and
                iteration == opt.simp_iteration1
                and opt.pruning_fraction1 > 0
            ):
                mask = get_pruning_mask(
                    gaussians._opacity[:, 0], opt.pruning_fraction1
                )
                gaussians.prune_points_admm(mask)
                if admm is not None:
                    admm.prune(mask)
            if (
                not args.semantic_only
                and
                iteration == opt.simp_iteration2
                and opt.pruning_fraction2 > 0
            ):
                mask = get_pruning_mask(
                    gaussians._opacity[:, 0], opt.pruning_fraction2
                )
                gaussians.prune_points_admm(mask)
                if admm is not None:
                    admm.prune(mask)

            if iteration < opt.iterations and iteration % args.accum_iter == 0:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            ema_total = 0.4 * loss.item() + 0.6 * ema_total
            ema_rgb = 0.4 * rgb_loss.item() + 0.6 * ema_rgb
            for level, level_loss in enumerate(semantic_losses):
                ema_semantic[level] = (
                    0.4 * level_loss.item() + 0.6 * ema_semantic[level]
                )
            if iteration % 10 == 0:
                display = {
                    "loss": "{:.5f}".format(ema_total),
                    "rgb": "{:.5f}".format(ema_rgb),
                    "points": gaussians.get_xyz.shape[0],
                }
                display.update({
                    "s{}".format(feature_levels[i]): "{:.5f}".format(value)
                    for i, value in enumerate(ema_semantic)
                })
                progress_bar.set_postfix(display)
                progress_bar.update(10)

            if tb_writer:
                tb_writer.add_scalar("loss/total", loss.item(), iteration)
                tb_writer.add_scalar("loss/rgb", rgb_loss.item(), iteration)
                tb_writer.add_scalar("loss/admm_opacity", opacity_admm_loss.item(), iteration)
                tb_writer.add_scalar("loss/admm_sh", sh_admm_loss.item(), iteration)
                tb_writer.add_scalar("scene/points", gaussians.get_xyz.shape[0], iteration)
                for index, level_loss in enumerate(semantic_losses):
                    tb_writer.add_scalar(
                        "loss/semantic_level_{}".format(feature_levels[index]),
                        level_loss.item(),
                        iteration,
                    )

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving joint checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(opt.include_feature), iteration),
                    os.path.join(scene.model_path, "chkpnt{}.pth".format(iteration)),
                )
                save_sh_quantization(admm, scene.model_path, iteration)

    progress_bar.close()
    if tb_writer:
        tb_writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="Joint multi-scale LangSplatV2 training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=55557)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--feature_levels", nargs=3, type=int, default=[1, 2, 3])
    parser.add_argument("--semantic_only", action="store_true", default=False)
    parser.add_argument("--reset_iteration", action="store_true", default=False)
    parser.add_argument("--semantic_loss", choices=("cos", "l1", "cos+l1"), default="cos")
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--accum_iter", type=int, default=1)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--no_rgb_loss", action="store_true", default=False)
    parser.add_argument("--no_language_loss", action="store_true", default=False)
    parser.add_argument("--no_admm_loss", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    args.save_iterations = sorted(set(args.save_iterations))
    args.checkpoint_iterations = sorted(set(args.checkpoint_iterations))
    print(args)
    print("Optimizing {}".format(args.model_path))

    safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args,
    )
    print("\nJoint training complete.")
