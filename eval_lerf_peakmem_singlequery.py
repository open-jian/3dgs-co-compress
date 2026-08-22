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

import torch.nn.functional as F


def save_lerf_metrics(args, chosen_iou_all, chosen_lvl_list, acc_num,
                      total_queries, timing=None):
    """Persist the exact values used by the LERF tables.

    The original evaluator only printed rounded metrics to a timestamped log,
    which makes aggregation error-prone.  Keep the log output for backwards
    compatibility and additionally write one machine-readable file per scene.
    """
    metrics = {
        "dataset": args.dataset_name,
        "checkpoint": (
            args.semantic_sidecar or args.compact_artifact
            or args.joint_checkpoint or args.checkpoint
        ),
        "mask_threshold": float(args.mask_thresh),
        "topk": int(args.topk),
        "rendered_miou": float(sum(chosen_iou_all) / len(chosen_iou_all)),
        "localization_correct": int(acc_num),
        "localization_total": int(total_queries),
        "localization_accuracy": float(acc_num / total_queries),
        "per_query_iou": [float(value) for value in chosen_iou_all],
        "chosen_semantic_level": [int(value) for value in chosen_lvl_list],
    }
    if args.semantic_sidecar:
        metrics["semantic_sidecar"] = os.path.abspath(args.semantic_sidecar)
        metrics["decoded_geometry_ply"] = os.path.abspath(args.geometry_ply)
        metrics["deployment_geometry_source"] = (
            "transient FCGS decode output; dense RGB checkpoint not loaded"
        )
    if timing is not None:
        metrics["timing"] = timing
    metrics_path = os.path.join(args.output_path, "metrics_lerf.json")
    with open(metrics_path, "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, sort_keys=True)
    logger.info("metrics json: %s", metrics_path)
    return metrics


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

def eval_gt_lerfdata(json_folder: Union[str, Path] = None, ouput_path: Path = None) -> Dict:
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
        idx = int(gt_data['info']['name'].split('_')[-1].split('.jpg')[0]) - 1
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

def segmentation_process_cuda(sem_map:torch.tensor, clip_model, thresh, img_ann,
                              prompts, visual_dir=None, rgb_img=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_map = clip_model.get_max_across_quick(sem_map)
    n_head, n_prompt, h, w = valid_map.shape

    # positive prompts
    chosen_iou_list, chosen_lvl_list = [], []
    iou_all = {}
    for k in range(n_prompt):
        iou_lvl = torch.zeros(n_head).to(device)
        mask_lvl = torch.zeros((n_head, h, w)).to(device)
        normalized_lvl = torch.zeros((n_head, h, w)).to(device)
        for i in range(n_head):
            scale = 29
            avg_pool = torch.nn.AvgPool2d(kernel_size=scale, stride=1, padding=14, count_include_pad=False).to(device)
            avg_filtered = avg_pool(valid_map[i][k].unsqueeze(0).unsqueeze(0))
            valid_map[i][k] = 0.5 * (avg_filtered.squeeze(0).squeeze(0) + valid_map[i][k])

            # truncate the heatmap into mask
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * (1.0 - (-1.0)) + (-1.0)
            output = torch.clip(output, 0, 1)
            normalized_lvl[i] = output

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
            score_lvl[i] = score
        chosen_lvl = torch.argmax(score_lvl)

        chosen_iou_list.append(iou_lvl[chosen_lvl].cpu().numpy().item())
        chosen_lvl_list.append(chosen_lvl.cpu().numpy().item())

        if visual_dir is not None:
            visual_dir = Path(visual_dir)
            visual_dir.mkdir(exist_ok=True, parents=True)
            safe_prompt = "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in prompts[k]
            )
            level = int(chosen_lvl.item())
            heat = (255.0 * normalized_lvl[level].detach().cpu().numpy()).astype(np.uint8)
            heat_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
            cv2.imwrite(str(visual_dir / f"{safe_prompt}_heatmap.png"), heat_bgr)
            mask = (255 * mask_lvl[level].detach().cpu().numpy()).astype(np.uint8)
            cv2.imwrite(str(visual_dir / f"{safe_prompt}_mask.png"), mask)
            if rgb_img is not None:
                rgb = rgb_img.detach().cpu().numpy()
                rgb_u8 = np.clip(255.0 * rgb, 0, 255).astype(np.uint8)
                heat_rgb = heat_bgr[..., ::-1]
                overlay = np.clip(0.55 * rgb_u8 + 0.45 * heat_rgb, 0, 255).astype(np.uint8)
                cv2.imwrite(
                    str(visual_dir / f"{safe_prompt}_overlay.png"),
                    overlay[..., ::-1],
                )

    return chosen_iou_list, chosen_lvl_list

def localization_process_cuda(sem_map:torch.tensor, clip_model, img_ann):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_map = clip_model.get_max_across_quick(sem_map)
    n_head, n_prompt, h, w = valid_map.shape

    # positive prompts
    select_level, scores_all = {}, {}
    acc_num = 0
    positives = list(img_ann.keys())
    for k in range(n_prompt):
        select_output = valid_map[:, k]
        scale = 29
        avg_pool = torch.nn.AvgPool2d(kernel_size=scale, stride=1, padding=14, count_include_pad=False).to(device)
        avg_filtered = avg_pool(select_output.unsqueeze(1)).squeeze(1)

        score_lvl = torch.zeros((n_head,))
        coord_lvl = []
        for i in range(n_head):
            score = avg_filtered[i].max()
            coord = torch.nonzero((avg_filtered[i] == score).type(torch.uint8))
            score_lvl[i] = score
            coord_lvl.append(coord)

        selec_head = torch.argmax(score_lvl)
        coord_final = coord_lvl[selec_head]

        scores_all[positives[k]] = score_lvl.tolist()
        select_level[positives[k]] = selec_head.item()

        for box in img_ann[positives[k]]['bboxes'].reshape(-1, 4):
            flag = 0
            x1, y1, x2, y2 = box
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            for cord_list in coord_final:
                if (cord_list[1] >= x_min and cord_list[1] <= x_max and
                    cord_list[0] >= y_min and cord_list[0] <= y_max):
                    acc_num += 1
                    flag = 1
                    break
            if flag != 0:
                break
    return acc_num

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
    gt_ann, image_shape, image_paths = eval_gt_lerfdata(Path(args.json_folder), Path(args.output_path))
    eval_index_list = [int(idx) for idx in list(gt_ann.keys())]
    clip_model = OpenCLIPNetwork(device)

    chosen_iou_all, chosen_lvl_list = [], []
    acc_num = 0

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
            language_feature_image = language_feature_image.permute(1, 2, 0)
            sem_feat.append(language_feature_image)

        restored_feat = torch.stack(sem_feat, dim=0)
        img_ann = gt_ann[f'{idx}']
        clip_model.set_positives(list(img_ann.keys()))

        c_iou_list, c_lvl = segmentation_process_cuda(
            restored_feat, clip_model, args.mask_thresh, img_ann,
            list(img_ann.keys()),
            image_name / "predictions" if args.save_visuals else None,
            rgb_img,
        )
        chosen_iou_all.extend(c_iou_list)
        chosen_lvl_list.extend(c_lvl)
        acc_num_img = localization_process_cuda(restored_feat, clip_model, img_ann)
        acc_num += acc_num_img

    logger.info(f'checkpoint: {args.checkpoint}')
    mean_iou_chosen = sum(chosen_iou_all) / len(chosen_iou_all)
    logger.info(f'trunc thresh: {args.mask_thresh}')
    logger.info(f"iou chosen: {mean_iou_chosen:.4f}")
    logger.info(f"chosen_lvl: \n{chosen_lvl_list}")

    # localization acc
    total_bboxes = 0
    for img_ann in gt_ann.values():
        total_bboxes += len(list(img_ann.keys()))
    acc = acc_num / total_bboxes
    logger.info("Localization accuracy: " + f'{acc:.4f}')

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
    gt_ann, image_shape, image_paths = eval_gt_lerfdata(Path(args.json_folder), Path(args.output_path))
    eval_index_list = [int(idx) for idx in list(gt_ann.keys())]
    clip_model = OpenCLIPNetwork(device)

    chosen_iou_all, chosen_lvl_list = [], []
    acc_num = 0

    load_start = time.perf_counter()
    combined_gaussians = GaussianModel(dataset.sh_degree)
    dataset.model_path = (
        os.path.dirname(os.path.abspath(args.semantic_sidecar))
        if args.semantic_sidecar else (
            os.path.dirname(os.path.abspath(args.compact_artifact))
            if args.compact_artifact else (
                os.path.dirname(os.path.abspath(args.joint_checkpoint))
                if args.joint_checkpoint else args.ckpt_paths[0]
            )
        )
    )
    scene = Scene(dataset, combined_gaussians, shuffle=False)
    views = scene.getTrainCameras()
    if args.semantic_sidecar:
        from semantic_sidecar import load_sidecar_into_gaussians
        # The PLY must be the transient output of decoding the counted FCGS
        # bitstream with its declared shared checkpoint.  Do not load the
        # dense RGB training/import checkpoint during deployment evaluation.
        combined_gaussians.load_ply(args.geometry_ply)
        combined_gaussians, _sidecar_bundle = load_sidecar_into_gaussians(
            combined_gaussians, args.semantic_sidecar, device="cuda"
        )
    elif args.compact_artifact:
        from compact_artifact import load_compact_gaussians
        combined_gaussians, _compact_bundle = load_compact_gaussians(
            args.compact_artifact, device="cuda"
        )
    else:
        checkpoint = (
            args.joint_checkpoint
            if args.joint_checkpoint
            else os.path.join(args.ckpt_paths[0], f'chkpnt{args.checkpoint}.pth')
        )
        (model_params, first_iter) = torch.load(checkpoint)
        combined_gaussians.restore(model_params, args, mode='test')
    if args.joint_checkpoint:
        combined_gaussians.prepare_multiscale_quick_render(topk=args.topk)
    elif not args.compact_artifact and not args.semantic_sidecar:
        language_feature_weights = []
        language_feature_indices = []
        language_feature_codebooks = []
        for level_idx in range(3):
            gaussians = GaussianModel(dataset.sh_degree)
            checkpoint = os.path.join(args.ckpt_paths[level_idx], f'chkpnt{args.checkpoint}.pth')
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, args, mode='test')
            language_feature_codebooks.append(gaussians._language_feature_codebooks.view(-1, 512))
            weights, indices = get_weights_and_indices(gaussians._language_feature_logits, args.topk)
            language_feature_weights.append(weights)
            language_feature_indices.append(indices + int(level_idx * gaussians._language_feature_codebooks.shape[1]))
        combined_gaussians._language_feature_codebooks = torch.stack(language_feature_codebooks, dim=0)
        combined_gaussians._language_feature_weights = torch.cat(language_feature_weights, dim=1)
        combined_gaussians._language_feature_indices = torch.cat(language_feature_indices, dim=1)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    representation_load_seconds = time.perf_counter() - load_start

    # Keep the deployed representation resident, but exclude checkpoint and
    # artifact deserialization from the inference-memory boundary.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_memory_allocated_bytes = torch.cuda.memory_allocated()
    baseline_memory_reserved_bytes = torch.cuda.memory_reserved()
    render_seconds = []
    query_seconds = []

    for i, idx in enumerate(tqdm(eval_index_list)):
        rgb_img = cv2.imread(image_paths[i])[..., ::-1]
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img).to(device)

        image_name = Path(args.output_path) / f'{idx+1:0>5}'
        image_name.mkdir(exist_ok=True, parents=True)

        view = views[idx]
        img_ann = gt_ann[f'{idx}']
        # Match the other Table-1 profilers: one positive text query at a time.
        # The previous profiler encoded and scored every label in an annotated
        # view as one batch, so its peak was not comparable to the single-query
        # peaks reported for the other methods.
        for prompt in img_ann.keys():
            single_ann = {prompt: img_ann[prompt]}
            query_start = time.perf_counter()
            clip_model.set_positives([prompt])
            torch.cuda.synchronize()
            query_embedding_seconds = time.perf_counter() - query_start
            render_start = time.perf_counter()
            language_feature_image = render_language_feature_map_quick(
                combined_gaussians, view, pipeline, background, args
            )
            torch.cuda.synchronize()
            render_seconds.append(time.perf_counter() - render_start)
            restored_feat = language_feature_image.permute(0, 2, 3, 1)
            query_start = time.perf_counter()
            c_iou_list, c_lvl = segmentation_process_cuda(
                restored_feat, clip_model, args.mask_thresh, single_ann,
                [prompt],
                image_name / "predictions" if args.save_visuals else None,
                rgb_img,
            )
            chosen_iou_all.extend(c_iou_list)
            chosen_lvl_list.extend(c_lvl)
            acc_num_img = localization_process_cuda(
                restored_feat, clip_model, single_ann
            )
            torch.cuda.synchronize()
            query_seconds.append(
                query_embedding_seconds + time.perf_counter() - query_start
            )
            acc_num += acc_num_img
            del restored_feat, language_feature_image
            torch.cuda.empty_cache()
        del rgb_img

    logger.info(f'checkpoint: {args.checkpoint}')
    mean_iou_chosen = sum(chosen_iou_all) / len(chosen_iou_all)
    logger.info(f'trunc thresh: {args.mask_thresh}')
    logger.info(f"iou chosen: {mean_iou_chosen:.4f}")
    logger.info(f"chosen_lvl: \n{chosen_lvl_list}")

    # localization acc
    total_bboxes = 0
    for img_ann in gt_ann.values():
        total_bboxes += len(list(img_ann.keys()))
    acc = acc_num / total_bboxes
    logger.info("Localization accuracy: " + f'{acc:.4f}')

    timing = {
        "representation_load_seconds": float(representation_load_seconds),
        "semantic_render_ms_per_view": float(1000.0 * sum(render_seconds) / len(render_seconds)),
        "query_batch_ms_per_view": float(1000.0 * sum(query_seconds) / len(query_seconds)),
        "query_ms_per_prompt": float(1000.0 * sum(query_seconds) / total_bboxes),
        "annotated_views": int(len(eval_index_list)),
        "memory": {
            "scope": "resident deployment plus one complete single-positive LERF query path; maximum across annotated view-query pairs",
            "baseline_memory_allocated_bytes": int(baseline_memory_allocated_bytes),
            "baseline_memory_reserved_bytes": int(baseline_memory_reserved_bytes),
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
    }
    save_lerf_metrics(
        args, chosen_iou_all, chosen_lvl_list, acc_num, total_bboxes, timing
    )

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
    parser.add_argument("--compact_artifact", default=None, type=str)
    parser.add_argument("--geometry_ply", default=None, type=str)
    parser.add_argument("--semantic_sidecar", default=None, type=str)
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
    parser.add_argument("--save_visuals", action="store_true")
    #------------------------------------------------------------

    args = get_combined_args(parser)
    # ``get_combined_args`` drops command-line options whose value is None
    # when they are absent from an older cfg_args file.  Deployment selectors
    # are intentionally optional, so restore their explicit defaults here.
    for optional_name in (
        "joint_checkpoint", "compact_artifact", "geometry_ply",
        "semantic_sidecar",
    ):
        if not hasattr(args, optional_name):
            setattr(args, optional_name, None)
    if bool(args.geometry_ply) != bool(args.semantic_sidecar):
        raise ValueError(
            "--geometry_ply and --semantic_sidecar must be provided together"
        )
    deployment_inputs = sum(bool(value) for value in (
        args.joint_checkpoint, args.compact_artifact, args.semantic_sidecar
    ))
    if deployment_inputs > 1:
        raise ValueError(
            "choose one of joint checkpoint, compact artifact, or FCGS semantic sidecar"
        )
    if deployment_inputs and not args.quick_render:
        raise ValueError("joint/compact/sidecar evaluation requires --quick_render")
    if args.semantic_sidecar:
        # Skip the legacy RGB optimizer reload and enable semantic rendering.
        args.include_feature = True
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
