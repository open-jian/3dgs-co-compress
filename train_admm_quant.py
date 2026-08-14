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
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene_admm import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments_admm_quant import ModelParams, PipelineParams, OptimizationParams
from admm import ADMM, get_pruning_mask
from datetime import datetime
from logger import get_logger
from quant import *
import yaml
import pprint
from omegaconf import OmegaConf


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
def workspace_output_root():
    return os.path.abspath(
        os.environ.get("OUTPUT_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "Output"))
    )


def save_features_to_npy(language_feature, gt_language_feature, mask, output_root, image_name="sample"):
    debug_dir = os.path.join(output_root, "debug_features")
    os.makedirs(debug_dir, exist_ok=True)
    print(f"shape of language_feature: {language_feature.shape}, gt_language_feature: {gt_language_feature.shape}, mask: {mask.shape}")
    # use mask
    masked_pred = (language_feature * mask).detach().cpu().numpy()
    masked_gt = (gt_language_feature * mask).detach().cpu().numpy()

    # Construct save path
    np.save(os.path.join(debug_dir, f"{image_name}_masked_pred.npy"), masked_pred)
    np.save(os.path.join(debug_dir, f"{image_name}_masked_gt.npy"), masked_gt)

    print(f"Saved debug features to {debug_dir}")

def training(dataset, opt : OptimizationParams, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, logger):
    checkpoint_iterations.append(opt.iterations)
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # set admm = True to ensure every fatures are in the backpropgation
    gaussian = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussian)
    gaussian.training_setup(opt)

    if opt.include_feature:
        if not checkpoint:
            raise ValueError("checkpoint missing!!!!!")
    if checkpoint:
        (model_params, _) = torch.load(checkpoint)
        gaussian.restore(model_params, opt)
        
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)


    # k-Means quantization
    quantized_params = opt.quant_params
    n_cls = opt.kmeans_ncls
    n_cls_sh = opt.kmeans_ncls_sh
    n_cls_dc = opt.kmeans_ncls_dc

    kmeans_rot_q = None  # or some default value
    kmeans_dc_q = None
    kmeans_sh_q = None
    kmeans_sc_q = None

    kmeans_f1_q = None
    kmeans_f2_q = None
    kmeans_f3_q = None
    if 'pos' in quantized_params:
        kmeans_pos_q = Quantize_kMeans(num_clusters=n_cls_dc)
    if 'dc' in quantized_params:
        kmeans_dc_q = Quantize_kMeans(num_clusters=n_cls_dc)
    if 'sh' in quantized_params:
        kmeans_sh_q = Quantize_kMeans(num_clusters=n_cls_sh)
    if 'scale' in quantized_params:
        kmeans_sc_q = Quantize_kMeans(num_clusters=n_cls)
    if 'rot' in quantized_params:
        kmeans_rot_q = Quantize_kMeans(num_clusters=n_cls)
    if 'scale_rot' in quantized_params:
        kmeans_scrot_q = Quantize_kMeans(num_clusters=n_cls)
    if 'sh_dc' in quantized_params:
        kmeans_shdc_q = Quantize_kMeans(num_clusters=n_cls_sh)
    if 'language_feature1' in quantized_params:
        kmeans_f1_q = Quantize_kMeans(num_clusters=n_cls)
    if 'language_feature2' in quantized_params:
        kmeans_f2_q = Quantize_kMeans(num_clusters=n_cls)
    if 'language_feature3' in quantized_params:
        kmeans_f3_q = Quantize_kMeans(num_clusters=n_cls)


    viewpoint_stack = None

    ema_loss_for_log = 0.0
    ema_admm1_for_log = 0.0
    ema_admm2_for_log = 0.0
    ema_language_for_log = 0.0
    ema_language2_for_log = 0.0
    ema_language3_for_log = 0.0
    ema_rgb_for_log = 0.0
    ema_quant_loss_for_log = 0.0

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
                    net_image = render(custom_cam, gaussian, pipe, background, opt, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        admm_loss1 = torch.tensor(0.0)
        admm_loss2 = torch.tensor(0.0)

        # Quant Module initialization
        if iteration >= (opt.quant_start_iter-3) and iteration < opt.admm_start_iter:
            logger.info("|K-means| Init...")
            kmeans_sh_q.forward_frest(gaussian, assign=False, update_centers_flag=True)

        with torch.no_grad():
            # ADMM Module initialization
            if iteration >= opt.admm_start_iter and iteration <= opt.simp_iteration2:
                if iteration == opt.admm_start_iter:
                    logger.info("|ADMM| Init...")
                    admm = ADMM(gaussian, opt.rho1, opt.rho2, device="cuda")
                    admm.update1(opt.pruning_threshold2, update_u=False)
                    admm.update2(kmeans_sh_q, update_u=False, assign = True, update_center = True)
                elif iteration % opt.admm_interval == 0:  
                    admm.update1(opt.pruning_threshold2)  
                    if iteration > opt.quant_start_iter and  iteration < opt.freeze_center_iter:
                        # 某个iter以后不再更新center
                        admm.update2(kmeans_sh_q, update_u=False, assign = True, update_center = False)
                    else:
                        admm.update2(kmeans_sh_q, update_u=False, assign = True, update_center = True) 


        gaussian.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussian.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussian, pipe, background, opt)
        image, language_feature, language_feature2, language_feature3= render_pkg["render"], render_pkg["language_feature_image1"], render_pkg["language_feature_image2"], render_pkg["language_feature_image3"]
        Ll1_1 = Ll1_2 = Ll1_3 = 0

        # Loss
        gt_language_feature, language_feature_mask = viewpoint_cam.get_language_feature(language_feature_dir=dataset.lf_path, feature_level=1)
        gt_language_feature2, language_feature_mask2 = viewpoint_cam.get_language_feature(language_feature_dir=dataset.lf_path, feature_level=2)
        gt_language_feature3, language_feature_mask3 = viewpoint_cam.get_language_feature(language_feature_dir=dataset.lf_path, feature_level=3)
        if viewpoint_cam.image_name in ["frame_00041","frame_00105","frame_00152","frame_00195"]:
            save_features_to_npy(language_feature, gt_language_feature, language_feature_mask, scene.model_path, image_name=f"iter_{iteration}_{viewpoint_cam.image_name}_lv1")
            save_features_to_npy(language_feature, gt_language_feature, language_feature_mask, scene.model_path, image_name=f"iter_{iteration}_{viewpoint_cam.image_name}_lv2")
            save_features_to_npy(language_feature, gt_language_feature, language_feature_mask, scene.model_path, image_name=f"iter_{iteration}_{viewpoint_cam.image_name}_lv3")

        Ll1_1 = l1_loss(language_feature*language_feature_mask, gt_language_feature*language_feature_mask)            
        Ll1_2 = l1_loss(language_feature2*language_feature_mask2, gt_language_feature2*language_feature_mask2)            
        Ll1_3 = l1_loss(language_feature3*language_feature_mask3, gt_language_feature3*language_feature_mask3)    
        lang_loss = Ll1_1 + Ll1_2 + Ll1_3

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1_rgb = l1_loss(image, gt_image)
        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1_rgb + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss = torch.tensor(0.0, requires_grad=True).to("cuda")

        if opt.enable_lang_loss:
            loss = loss + lang_loss * opt.language_loss_coeff

        if opt.enable_rgb_loss:
            loss = loss + rgb_loss * opt.rgb_loss_coeff

        if opt.enable_admm_loss and iteration % opt.admm_interval == 0 and iteration >= opt.admm_start_iter and iteration <= opt.admm_end_iter:    
            admm_loss1 = admm.get_admm_loss_1()
            admm_loss2 = admm.get_admm_loss_2()
            admm_loss = admm_loss1 + admm_loss2
            loss = loss + admm_loss * opt.admm_loss_coeff

        loss.backward()

        iter_end.record()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_admm1_for_log = 0.4 * admm_loss1.item() + 0.6 * ema_admm1_for_log
            ema_admm2_for_log = 0.4 * admm_loss2.item() + 0.6 * ema_admm2_for_log
            ema_rgb_for_log = 0.4 * rgb_loss.item() + 0.6 * ema_loss_for_log
            ema_language_for_log = 0.4 * Ll1_1.item() + 0.6 * ema_language_for_log
            ema_language2_for_log = 0.4 * Ll1_2.item() + 0.6 * ema_language2_for_log
            ema_language3_for_log = 0.4 * Ll1_3.item() + 0.6 * ema_language3_for_log

            if not opt.enable_rgb_loss:
                ema_rgb_for_log = 0

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {"Loss": f"{ema_loss_for_log:.{4}f}", 
                     "rgb": f"{ema_rgb_for_log:.{4}f}", 
                     "Ll1": f"{ema_language_for_log:.{4}f}",
                     "Ll2": f"{ema_language2_for_log:.{4}f}",
                     "Ll3": f"{ema_language3_for_log:.{4}f}",
                     "admm1": f"{admm_loss1.item() :.{4}f}", 
                     "admm2": f"{admm_loss2.item() :.{4}f}", 
                     "Points": gaussian._xyz.shape[0]})
                progress_bar.update(10)
                log_info = {"iter":iteration,
                            "loss": round(ema_loss_for_log, 7),
                            "rgb": round(ema_rgb_for_log, 7),
                            "Ll1": round(ema_language_for_log, 7),
                            "Ll2": round(ema_language2_for_log, 7),
                            "Ll3": round(ema_language3_for_log, 7),
                            "admm1": round(ema_admm1_for_log, 7),
                            "admm2": round(ema_admm2_for_log, 7),
                            "point": gaussian._xyz.shape[0],
                        }
                logger.info(log_info)


            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, opt))
            # if (iteration in saving_iterations):
            #     logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration)



            if opt.prune and (iteration == opt.simp_iteration1) and opt.pruning_threshold1>0:
                scores = gaussian._opacity[:, 0]
                mask = get_pruning_mask(scores, opt.pruning_threshold1)
                gaussian.prune_points_admm(mask)

            if opt.prune and (iteration == opt.simp_iteration2) and opt.pruning_threshold2>0:
                scores = gaussian._opacity[:, 0]
                mask = get_pruning_mask(scores, opt.pruning_threshold2)
                gaussian.prune_points_admm(mask) 
                if opt.quant:
                    kmeans_sh_q.prune(mask)  

            # Optimizer step
            if iteration < opt.iterations:
                gaussian.optimizer.step()
                gaussian.optimizer.zero_grad(set_to_none = True)






    
    kmeans_sh_q.cluster_assign(gaussian._features_rest)
    deg = gaussian._features_rest.shape[1]
    sampled_centers = torch.gather(kmeans_sh_q.centers, 0, kmeans_sh_q.nn_index.unsqueeze(-1).repeat(1, kmeans_sh_q.vec_dim))
    gaussian._features_rest_q = sampled_centers.reshape(-1, deg, 3).detach()

    print("\n[ITER {}] Training complete".format(iteration))

    all_attributes = {'xyz': 'xyz', 
                        'dc': 'f_dc', 
                        'sh': 'f_rest', 
                        'opacities': 'opacities',
                        'scale': 'scale', 
                        'rot': 'rotation',
                        'language_feature1': 'language_feature1',
                        'language_feature2': 'language_feature2',
                        'language_feature3': 'language_feature3'}
    if args.quant:
        # Unquantized parameters
        non_quant_attributes = [val for (key, val) in all_attributes.items() if key not in quantized_params]
        scene.save_non_quant_attributes(iteration, save_attributes=non_quant_attributes)

        # Quantized parameters
        print('quantized attributes: ', quantized_params)
        kmeans_dict = {'rot': kmeans_rot_q, 'scale': kmeans_sc_q, 'sh': kmeans_sh_q, 'dc': kmeans_dc_q, 'language_feature1': kmeans_f1_q, 'language_feature2': kmeans_f2_q, 'language_feature3': kmeans_f3_q}
        kmeans_list = []
        for param in quantized_params:
            kmeans_list.append(kmeans_dict[param])
        out_dir = join(scene.model_path, 'point_cloud/iteration_%d' % iteration)
        save_kmeans(kmeans_list, quantized_params, out_dir)
    else:
        scene.save_non_quant_attributes(iteration, save_attributes=all_attributes.values())

    if (iteration in checkpoint_iterations):
        torch.save((gaussian.capture(opt.include_feature), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        logger.info(f"[checkpoint_iterations {iteration}] Saving Gaussians in {scene.model_path}")


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join(workspace_output_root(), "colasplat", "admm_quant", "runs", unique_str[0:10])
        
    # Set up output folder
    logger.info("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    # tb_writer = None
    # if TENSORBOARD_FOUND:
    #     tb_writer = SummaryWriter(args.model_path)
    # else:
    #     print("Tensorboard not available: not logging progress")
    # return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        logger.info(f'testing for iter {iteration}')
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
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()



def load_config_with_scene(dataset_name, scene_name):
    base_path = f"configs/{dataset_name}"
    base_config_path = os.path.join(base_path, "base.yaml")
    scene_config_path = os.path.join(base_path, f"{scene_name}.yaml")


    base_cfg = OmegaConf.load(base_config_path)
    
    if os.path.exists(scene_config_path):
        scene_cfg = OmegaConf.load(scene_config_path)
        config = OmegaConf.merge(base_cfg, scene_cfg)
        print(f"[INFO] Loaded base.yaml + {scene_name}.yaml (scene config overrides base)")
    else:
        config = base_cfg
        print(f"[INFO] Loaded base.yaml (no scene-specific config found)")

    return config

def inject_config_into_group(config_dict, opt):
    for k, v in config_dict.items():
        setattr(opt, k, v)
    return opt


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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)
    args.model_path = args.model_path
    print("Output:" + args.model_path)
    scene_name = os.path.basename(args.source_path.rstrip("/"))
    logger = get_logger(
        scene_name,
        os.path.join(workspace_output_root(), "colasplat", "logs", "train_admm_quant", scene_name),
    )

    # dataset_name = "lerf" #3dovs
    # config = load_config_with_scene(dataset_name, scene_name)
    # args = inject_config_into_group(config, args)
    # logger.info("Config loaded:\n%s", config)

    logger.info("Train_admm_quant started at {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    args_dict = vars(args)  # 
    print(f"args:\n{pprint.pformat(args_dict)}")
    logger.info("args:\n%s", pprint.pformat(args_dict))




    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, logger)
    


    # All done
    logger.info("\nTraining complete.")
