#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import torch
from random import randint
from utils.loss_utils import l1_loss,  ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from datetime import datetime
from logger import get_logger

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import numpy as np
    
def workspace_output_root():
    return os.path.abspath(
        os.environ.get("OUTPUT_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "Output"))
    )


def save_features_to_npy(language_feature, gt_language_feature, mask, output_root, image_name="sample"):
    debug_dir = os.path.join(output_root, "debug_features")
    os.makedirs(debug_dir, exist_ok=True)
    print(f"shape of language_feature: {language_feature.shape}, gt_language_feature: {gt_language_feature.shape}, mask: {mask.shape}")

    masked_pred = (language_feature * mask).detach().cpu().numpy()
    masked_gt = (gt_language_feature * mask).detach().cpu().numpy()


    np.save(os.path.join(debug_dir, f"{image_name}_masked_pred.npy"), masked_pred)
    np.save(os.path.join(debug_dir, f"{image_name}_masked_gt.npy"), masked_gt)

    print(f"Saved debug features to {debug_dir}")

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, logger):
    checkpoint_iterations.append(opt.iterations)
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if opt.include_feature:
        if not checkpoint:
            raise ValueError("checkpoint missing!!!!!")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        if len(model_params) == 12 and opt.include_feature:
            first_iter = 0
        gaussians.restore(model_params, opt)
        
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_loss_level1 = 0.0
    ema_loss_level2 = 0.0
    ema_loss_level3 = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, opt, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, opt)
        image, language_feature1, language_feature2, language_feature3, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["language_feature_image1"], render_pkg["language_feature_image2"], render_pkg["language_feature_image3"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        Ll1 = Ll1_2 = Ll1_3 = 0
        # Loss
        if opt.include_feature:
            gt_language_feature1, language_feature_mask1 = viewpoint_cam.get_language_feature(language_feature_dir=dataset.lf_path, feature_level=1)
            gt_language_feature2, language_feature_mask2 = viewpoint_cam.get_language_feature(language_feature_dir=dataset.lf_path, feature_level=2)
            gt_language_feature3, language_feature_mask3 = viewpoint_cam.get_language_feature(language_feature_dir=dataset.lf_path, feature_level=3)

            save_features_to_npy(
                language_feature,
                gt_language_feature,
                language_feature_mask,
                scene.model_path,
                image_name=viewpoint_cam.image_name,
            )

            Ll1 = l1_loss(language_feature1*language_feature_mask1, gt_language_feature1*language_feature_mask1)            
            Ll1_2 = l1_loss(language_feature2*language_feature_mask2, gt_language_feature2*language_feature_mask2)            
            Ll1_3 = l1_loss(language_feature3*language_feature_mask3, gt_language_feature3*language_feature_mask3)            

            loss = Ll1 + Ll1_2 + Ll1_3
        else:
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()
        iter_end.record()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_loss_level1 = 0.4 * Ll1.item() + 0.6 * ema_loss_level1
            ema_loss_level2 = 0.4 * Ll1_2.item() + 0.6 * ema_loss_level2
            ema_loss_level3 = 0.4 * Ll1_3.item() + 0.6 * ema_loss_level3

            
            if iteration % 10 == 0:
                # progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}","lv1": f"{ema_loss_level1:.{7}f}","lv2": f"{ema_loss_level2:.{7}f}","lv3": f"{ema_loss_level3:.{7}f}"})
                # progress_bar.update(10)
                log_info = {"iter":iteration,
                            "loss": round(ema_loss_for_log, 7),
                            "lv1": round(ema_loss_level1, 7),
                            "lv2": round(ema_loss_level2, 7),
                            "lv3": round(ema_loss_level3, 7),
                            "point": gaussians._xyz.shape[0]
                        }
                logger.info(log_info)

            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, opt))
            if (iteration in saving_iterations):
                logger.info(f"Ply saved in iter {iteration}")
                scene.save(iteration)

            # Densification
            if not opt.include_feature:
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(opt.include_feature), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                logger.info(f"[checkpoint_iterations {iteration}] Saving Gaussians in {scene.model_path}")


    torch.save((gaussians.capture(opt.include_feature), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

 
    scene.save(iteration)
            
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join(workspace_output_root(), "colasplat", "runs", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        print(f'testing for iter {iteration}')
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # if tb_writer:
        #     tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=55555)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int,  default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print(args)
    print("The output model (pth & ply) will saved in: " + args.model_path)


    # Initialize system state (RNG)
    safe_state(args.quiet)
    scene_name = os.path.basename(args.source_path.rstrip("/"))

    log_path = os.path.join(workspace_output_root(), "colasplat", "logs", "train", scene_name)
    logger = get_logger(scene_name, log_path)
    logger.info("Training started at {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, logger)


    # All done
    print("\nTraining complete.")
