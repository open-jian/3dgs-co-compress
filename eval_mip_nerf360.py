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
import numpy as np
import torch
import torchvision.transforms as transforms
import os
import random
from tqdm import tqdm
import time
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from pathlib import Path
import cv2
import logging

from eval.openclip_encoder import OpenCLIPNetwork
from scene import Scene
import eval.colormaps as colormaps
import json
import glob
from collections import defaultdict
from typing import Dict, Union
import sys
sys.path.append("eval")
from eval.utils import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result
import numpy as np
from utils.vq_utils import get_weights_and_indices

def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger

def eval_gt_mipdata(json_folder: Union[str, Path] = None, ouput_path: Path = None) -> Dict:
    """
    organise lerf's gt annotations
    gt format:
        file name: frame_xxxxx.json
        file content: labelme format
    return:
        gt_ann: dict()
            keys: str(int(idx))
            values: dict()
                keys: str(label)
                values: dict() which contain 'bboxes' and 'mask'
    """
    gt_json_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.json')))
    img_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.jpg')))
    gt_ann = {}
    for js_path in gt_json_paths:
        img_ann = defaultdict(dict)
        with open(js_path, 'r') as f:
            gt_data = json.load(f)
        
        h, w = gt_data['info']['height'], gt_data['info']['width']
        idx = int(gt_data['info']['name'].split('_')[-1].split('.jpg')[0]) - 1 # name-1
        for prompt_data in gt_data["objects"]:
            label = prompt_data['category']
            box = np.asarray(prompt_data['bbox']).reshape(-1)           # x1y1x2y2
            mask = polygon_to_mask((h, w), prompt_data['segmentation'])
            if img_ann[label].get('mask', None) is not None:
                mask = stack_mask(img_ann[label]['mask'], mask)
                img_ann[label]['bboxes'] = np.concatenate(
                    [img_ann[label]['bboxes'].reshape(-1, 4), box.reshape(-1, 4)], axis=0)
            else:
                img_ann[label]['bboxes'] = box
            img_ann[label]['mask'] = mask
            
            # # save for visulsization
            save_path = ouput_path / 'gt' / gt_data['info']['name'].split('.jpg')[0] / f'{label}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            vis_mask_save(mask, save_path)
        gt_ann[f'{idx}'] = img_ann

    return gt_ann, (h, w), img_paths

def smooth_cuda(mask_pred:torch.Tensor):
    scale = 7
    avg_pool = torch.nn.AvgPool2d(kernel_size=scale, stride=1, padding=3, count_include_pad=False).to(mask_pred.device)
    avg_filtered = avg_pool(mask_pred.float().unsqueeze(0).unsqueeze(0))
    mask = (avg_filtered > 0.5).type(torch.uint8).squeeze(0).squeeze(0)
    return mask

def segmentation_process_cuda(sem_map:torch.tensor, clip_model, thresh, img_ann, prompts, image_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_map = clip_model.get_max_across_quick(sem_map)
    n_head, n_prompt, h, w = valid_map.shape

    # positive prompts
    chosen_iou_list, chosen_lvl_list = [], []
    iou_all = {}
    for k in range(n_prompt):
        iou_lvl = torch.zeros(n_head).to(device)
        mask_lvl = torch.zeros((n_head, h, w)).to(device)
        for i in range(n_head):
            scale = 29
            avg_pool = torch.nn.AvgPool2d(kernel_size=scale, stride=1, padding=14, count_include_pad=False).to(device)
            avg_filtered = avg_pool(valid_map[i][k].unsqueeze(0).unsqueeze(0))
            valid_map[i][k] = 0.5 * (avg_filtered.squeeze(0).squeeze(0) + valid_map[i][k])

            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * (1.0 - (-1.0)) + (-1.0)
            output = torch.clip(output, 0, 1)

            mask_pred = (output > thresh).type(torch.uint8)
            mask_pred = smooth_cuda(mask_pred)
            mask_lvl[i] = mask_pred
            mask_gt = torch.from_numpy(img_ann[prompts[k]]['mask'].astype(np.uint8)).to(device)

            # calculate iou
            intersection = torch.sum(torch.logical_and(mask_gt, mask_pred))
            union = torch.sum(torch.logical_or(mask_gt, mask_pred))
            iou = torch.sum(intersection) / torch.sum(union)
            iou_lvl[i] = iou

        iou_all[prompts[k]] = iou_lvl.tolist()
        score_lvl = torch.zeros((n_head,), device=valid_map.device)
        for i in range(n_head):
            score = valid_map[i, k].max()
            score = (valid_map[i, k] * mask_lvl[i]).sum() / mask_lvl[i].sum()
            score_lvl[i] = score
        chosen_lvl = torch.argmax(score_lvl)
        
        chosen_iou_list.append(iou_lvl[chosen_lvl].cpu().numpy().item())
        chosen_lvl_list.append(chosen_lvl.cpu().numpy().item())

        save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
        mask = mask_lvl[chosen_lvl].cpu().numpy()
        vis_mask_save(mask, save_path)

    return chosen_iou_list, chosen_lvl_list

def render_language_feature_map(gaussians:GaussianModel, view, pipeline, background, args):
    with torch.no_grad():
        output = render(view, gaussians, pipeline, background, args)
        language_feature_weight_map = output['language_feature_weight_map']
        language_feature_map = gaussians.compute_final_feature_map(language_feature_weight_map)

    return language_feature_map

def render_language_feature_map_quick(gaussians:GaussianModel, view, pipeline, background, args):
    with torch.no_grad():
        output = render(view, gaussians, pipeline, background, args)
        language_feature_weight_map = output['language_feature_weight_map']
        D, H, W = language_feature_weight_map.shape
        language_feature_weight_map = language_feature_weight_map.view(3, 64, H, W).view(3, 64, H*W)
        language_codebooks = gaussians.get_quick_codebooks().permute(0, 2, 1)
        language_feature_map = torch.einsum('ldk,lkn->ldn', language_codebooks, language_feature_weight_map).view(3, 512, H, W)
        language_feature_map = language_feature_map / (language_feature_map.norm(dim=1, keepdim=True) + 1e-10)
    
    return language_feature_map


def evaluate(dataset:ModelParams, pipeline:PipelineParams, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )
    # load test data
    gt_ann, image_shape, image_paths = eval_gt_mipdata(Path(args.json_folder), Path(args.output_path))
    eval_index_list = [int(idx) for idx in list(gt_ann.keys())]
    clip_model = OpenCLIPNetwork(device)

    chosen_iou_all, chosen_lvl_list = [], []
    resize = transforms.Resize(image_shape)

    for i, idx in enumerate(tqdm(eval_index_list)):
        rgb_img = cv2.imread(image_paths[i])[..., ::-1]
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img).to(device)

        image_name = Path(args.output_path) / f'{idx+1:0>5}'
        image_name.mkdir(exist_ok=True, parents=True)
        img_ann = gt_ann[f'{idx}']
        clip_model.set_positives(list(img_ann.keys()))
        sem_feat = []
        for level_idx in range(3):
            # restore gaussian model
            dataset.model_path = args.ckpt_paths[level_idx]
            gaussians = GaussianModel(dataset.sh_degree)
            scene = Scene(dataset, gaussians, shuffle=False)
            views = scene.getTrainCameras()
            view = views[idx]
            bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            checkpoint = os.path.join(args.ckpt_paths[level_idx], f'chkpnt{args.checkpoint}.pth')
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, args, mode='test')
            
            language_feature_image = render_language_feature_map(gaussians, view, pipeline, background, args)
            language_feature_image = language_feature_image / (language_feature_image.norm(dim=0, keepdim=True) + 1e-10)
            language_feature_image = language_feature_image.detach()
            language_feature_image = resize(language_feature_image)
            language_feature_image = language_feature_image.permute(1, 2, 0)
            sem_feat.append(language_feature_image)

        restored_feat = torch.stack(sem_feat, dim=0)
        img_ann = gt_ann[f'{idx}']
        clip_model.set_positives(list(img_ann.keys()))
        
        c_iou_list, c_lvl = segmentation_process_cuda(restored_feat, clip_model, args.mask_thresh, img_ann, list(img_ann.keys()), image_name)
        chosen_iou_all.extend(c_iou_list)
        chosen_lvl_list.extend(c_lvl)

    logger.info(f'checkpoint: {args.checkpoint}')
    mean_iou_chosen = sum(chosen_iou_all) / len(chosen_iou_all)
    logger.info(f'trunc thresh: {args.mask_thresh}')
    logger.info(f"iou chosen: {mean_iou_chosen:.4f}")
    logger.info(f"chosen_lvl: \n{chosen_lvl_list}")

    return

def evaluate_quick(dataset:ModelParams, pipeline:PipelineParams, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )
    # load test data
    gt_ann, image_shape, image_paths = eval_gt_mipdata(Path(args.json_folder), Path(args.output_path))
    eval_index_list = [int(idx) for idx in list(gt_ann.keys())]
    clip_model = OpenCLIPNetwork(device)

    chosen_iou_all, chosen_lvl_list = [], []
    resize = transforms.Resize(image_shape)
    for i, idx in enumerate(tqdm(eval_index_list)):
        rgb_img = cv2.imread(image_paths[i])[..., ::-1]
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img).to(device)

        image_name = Path(args.output_path) / f'{idx+1:0>5}'
        image_name.mkdir(exist_ok=True, parents=True)

        combined_gaussians = GaussianModel(dataset.sh_degree)
        dataset.model_path = (
            os.path.dirname(os.path.abspath(args.joint_checkpoint))
            if args.joint_checkpoint else args.ckpt_paths[0]
        )
        scene = Scene(dataset, combined_gaussians, shuffle=False)
        views = scene.getTrainCameras()
        view = views[idx]
        checkpoint = (
            args.joint_checkpoint
            if args.joint_checkpoint
            else os.path.join(args.ckpt_paths[0], f'chkpnt{args.checkpoint}.pth')
        )
        (model_params, first_iter) = torch.load(checkpoint)
        combined_gaussians.restore(model_params, args, mode='test')
        img_ann = gt_ann[f'{idx}']
        clip_model.set_positives(list(img_ann.keys()))
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if args.joint_checkpoint:
            combined_gaussians.prepare_multiscale_quick_render(topk=4)
        else:
            language_feature_weights = []
            language_feature_indices = []
            language_feature_codebooks = []
            for level_idx in range(3):
                gaussians = GaussianModel(dataset.sh_degree)
                checkpoint = os.path.join(args.ckpt_paths[level_idx], f'chkpnt{args.checkpoint}.pth')
                (model_params, first_iter) = torch.load(checkpoint)
                gaussians.restore(model_params, args, mode='test')
                language_feature_codebooks.append(gaussians._language_feature_codebooks.view(-1, 512))
                weights, indices = get_weights_and_indices(gaussians._language_feature_logits, 4)
                language_feature_weights.append(weights)
                language_feature_indices.append(indices + int(level_idx * gaussians._language_feature_codebooks.shape[1]))
            combined_gaussians._language_feature_codebooks = torch.stack(language_feature_codebooks, dim=0)
            combined_gaussians._language_feature_weights = torch.cat(language_feature_weights, dim=1)
            combined_gaussians._language_feature_indices = torch.cat(language_feature_indices, dim=1)
        
        language_feature_image = render_language_feature_map_quick(combined_gaussians, view, pipeline, background, args)
        language_feature_image = language_feature_image / (language_feature_image.norm(dim=1, keepdim=True) + 1e-10)
        language_feature_image = torch.stack([resize(language_feature_image[level]) for level in range(3)], dim=0)
        restored_feat = language_feature_image.permute(0, 2, 3, 1)
        c_iou_list, c_lvl = segmentation_process_cuda(restored_feat, clip_model, args.mask_thresh, img_ann, list(img_ann.keys()), image_name)
        chosen_iou_all.extend(c_iou_list)
        chosen_lvl_list.extend(c_lvl)

    logger.info(f'checkpoint: {args.checkpoint}')
    mean_iou_chosen = sum(chosen_iou_all) / len(chosen_iou_all)
    logger.info(f'trunc thresh: {args.mask_thresh}')
    logger.info(f"iou chosen: {mean_iou_chosen:.4f}")
    logger.info(f"chosen_lvl: \n{chosen_lvl_list}")

    return

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    seed_num = 42
    seed_everything(seed_num)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    #------------------------------------------------------------
    # arguments for gaussian model
    parser.add_argument("--ckpt_root_path", default='output', type=str)
    parser.add_argument("--joint_checkpoint", default=None, type=str)
    parser.add_argument("--include_feature", action="store_true")
    parser.add_argument("--quick_render", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    #------------------------------------------------------------
    #------------------------------------------------------------
    # arguments for evaluation and output
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--index", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--json_folder", type=str, default=None)
    parser.add_argument("--mask_thresh", type=float, default=0.4)
    parser.add_argument("--checkpoint", type=int, default=10000)
    parser.add_argument("--topk", type=int, default=1)
    #------------------------------------------------------------

    args = get_combined_args(parser)
    if args.joint_checkpoint and not args.quick_render:
        raise ValueError("--joint_checkpoint currently requires --quick_render")
    args.ckpt_paths = [os.path.join(args.ckpt_root_path, args.dataset_name + f"_{args.index}_{level}") for level in [1, 2, 3]]
    args.output_path = os.path.join(args.output_dir, args.dataset_name + f"_{args.index}")
    args.json_folder = os.path.join(args.json_folder, args.dataset_name)
    
    os.makedirs(args.output_path, exist_ok=True)
    # NOTE logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.makedirs(args.output_path, exist_ok=True)
    log_file = os.path.join(args.output_path, f'{timestamp}.log')
    logger = get_logger(f'{args.dataset_name}', log_file=log_file, log_level=logging.INFO)
    
    safe_state(args.quiet)
    print(args)
    with torch.no_grad():
        if args.quick_render:
            evaluate_quick(model.extract(args), pipeline.extract(args), args)
        else:
            evaluate(model.extract(args), pipeline.extract(args), args)
