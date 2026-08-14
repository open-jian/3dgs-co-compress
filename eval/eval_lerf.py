#!/usr/bin/env python
from __future__ import annotations

import json
import os
import glob
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union
from argparse import ArgumentParser
import logging
import cv2
import numpy as np
import torch
import time
from tqdm import tqdm
from skimage.io import imsave
from datetime import datetime
import math

import sys
# 为了方便debug
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

sys.path.append("..")
from logger import get_logger
import colormaps
from autoencoder.model import Autoencoder
from openclip_encoder import OpenCLIPNetwork
from eval.utils import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result, show_result_2
SAVE_IMAGE = True
from scipy.ndimage import label


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
    print(f'len(gt_json_paths) in {json_folder}: {len(gt_json_paths)}')
    print(f'len(img_paths) in {json_folder}: {len(img_paths)}')
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
            if SAVE_IMAGE:
                vis_mask_save(mask, save_path)
        gt_ann[f'{idx}'] = img_ann

    return gt_ann, (h, w), img_paths


def activate_stream(sem_map, 
                    image, 
                    clip_model, 
                    image_name: Path = None,
                    img_ann: Dict = None, 
                    thresh : float = 0.5, 
                    colormap_options = None,
                    idx = None,
                    scene_name = None,
                    logger = None):
                    
    valid_map = clip_model.get_max_across(sem_map)                 # 3xkx832x1264
    n_head, n_prompt, h, w = valid_map.shape

    # positive prompts
    chosen_iou_list, chosen_lvl_list = [], []
    for k in range(n_prompt):
        iou_lvl = np.zeros(n_head)
        mask_lvl = np.zeros((n_head, h, w))
        score_lvl = torch.zeros((n_head,), device=valid_map.device)
        for i in range(n_head):
            # NOTE 加滤波结果后的激活值图中找最大值点
            scale = 30
            kernel = np.ones((scale,scale)) / (scale**2)
            np_relev = valid_map[i][k].cpu().numpy()
            avg_filtered = cv2.filter2D(np_relev, -1, kernel)
            avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
            valid_map[i][k] = 0.5 * (avg_filtered + valid_map[i][k])
            
            output_path_relev = image_name / 'heatmap' / f'{clip_model.positives[k]}_{i}'
            output_path_relev.parent.mkdir(exist_ok=True, parents=True)
            if SAVE_IMAGE:
                colormap_saving(valid_map[i][k].unsqueeze(-1), colormap_options,output_path_relev)
            
            # NOTE 与lerf一致，激活值低于0.5的认为是背景
            p_i = torch.clip(valid_map[i][k] - 0.5, 0, 1).unsqueeze(-1)
            valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
            mask = (valid_map[i][k] < thresh).squeeze()
            valid_composited[mask, :] = image[mask, :] * 0.3
            output_path_compo = image_name / 'composited' / f'{clip_model.positives[k]}_{i}'
            output_path_compo.parent.mkdir(exist_ok=True, parents=True)
            if SAVE_IMAGE:
                colormap_saving(valid_composited, colormap_options, output_path_compo)
            
            # truncate the heatmap into mask
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * (1.0 - (-1.0)) + (-1.0)
            output = torch.clip(output, 0, 1)

            mask_pred = (output.cpu().numpy() > thresh).astype(np.uint8)
            mask_pred = smooth(mask_pred)
            mask_lvl[i] = mask_pred

            heatmap = valid_map[i][k]
            masked_region = heatmap * torch.from_numpy(mask_pred).to(heatmap.device)   # 仅保留 mask 区域的值
            masked_region_path = image_name / 'masked_region' / f'{clip_model.positives[k]}_{i}'
            masked_region_path.parent.mkdir(exist_ok=True, parents=True)
            if SAVE_IMAGE:
                colormap_saving(masked_region.unsqueeze(-1), colormap_options, masked_region_path)
                
            mask_gt = img_ann[clip_model.positives[k]]['mask'].astype(np.uint8)
            if isinstance(mask_gt, torch.Tensor):
                mask_npy = mask_gt.cpu().numpy()  # 转换为 numpy 数组
            else:
                mask_npy = mask_gt
            mask_gt = mask_npy.astype(np.uint8)      
            mask_gt = (mask_gt > 0).astype(np.uint8)  # 非0视为前景

            if SAVE_IMAGE:
                # 保存 GT 掩码（可视化用）
                mask_gt_255 = (mask_gt.astype(np.uint8) * 255)  # 将True映射为255，False映射为0
                mask_pred_255 = (mask_pred.astype(np.uint8) * 255)
                mask_gt_path = image_name / 'mask_gt' / f'{clip_model.positives[k]}_{i}.png'
                mask_gt_path.parent.mkdir(exist_ok=True, parents=True)
                mask_pred_path = image_name / 'mask_pred' / f'{clip_model.positives[k]}_{i}.png'
                mask_pred_path.parent.mkdir(exist_ok=True, parents=True)

                imsave(mask_gt_path, mask_gt_255)
                imsave(mask_pred_path, mask_pred_255)


            # calculate iou
            intersection = np.sum(np.logical_and(mask_gt, mask_pred))
            union = np.sum(np.logical_or(mask_gt, mask_pred))
            iou = np.sum(intersection) / np.sum(union)
            iou_lvl[i] = iou


            heatmap = valid_map[i, k]
            # 低频结构比例（频域平滑度）
            fft = torch.fft.fft2(heatmap)
            fft_mag = fft.abs()
            high_freq_energy = fft_mag[20:, 20:].mean()  # 按你图像大小调整
            low_freq_ratio = -high_freq_energy* 0.03             # 惩罚高频（噪声）
            mean_val = heatmap.mean()
            max_val = heatmap.max()

            c_fft = torch.fft.fft2(masked_region)
            c_fft_mag = c_fft.abs()
            c_high_freq_energy = c_fft_mag[20:, 20:].mean()  # 按你图像大小调整
            c_low_freq_ratio = -c_high_freq_energy * 0.03       # 惩罚高频（噪声）
            c_mean_val = masked_region.mean() if masked_region.numel()  > 0 else 0
            c_max_val = masked_region.max() if masked_region.numel()  > 0 else 0
            #转为二值：
            binary_mask = (masked_region > 0).to(torch.uint8)
            # 计算连通区域（离散块）的数量
            structure = np.ones((3, 3), dtype=np.uint8)  # 8连通
            # num_regions是离散块的个数
            # 如果 binary_mask 是 tensor 且在GPU上
            binary_mask_cpu = binary_mask.cpu().numpy()
            labeled_mask, num_regions = label(binary_mask_cpu, structure=structure)
            L_frag = -math.exp(0.1 * num_regions) if num_regions > 3 else 0  # 指数级碎块惩罚

            suffix = f"{clip_model.positives[k]}{idx:0>5}"
            score = max_val
            logger.info(f"[{suffix}], score:{score:.4f}=max_val[{max_val:.4f}]/lfr[{low_freq_ratio:.4f}]/mean_val[{mean_val:.4f}]/cmax[{c_max_val:.4f}]/c_lfr[{c_low_freq_ratio:.4f}]/c_mean[{c_mean_val:.4f}]/frag[{num_regions}]/L_frag[{L_frag:.4f}]")

            score_lvl[i] = score



        # score_lvl = torch.zeros((n_head,), device=valid_map.device)
        # for i in range(n_head):
        #     score = valid_map[i, k].max()
        #     score_lvl[i] = score
        chosen_lvl = torch.argmax(score_lvl)
        suffix = f"{clip_model.positives[k]}{idx:0>5}"
        logger.info(f"[{suffix}], scores_list = {score_lvl}, choose[{chosen_lvl+1}], iou_list = {np.array2string(iou_lvl, precision=4)}")

        chosen_iou_list.append(iou_lvl[chosen_lvl])
        chosen_lvl_list.append(chosen_lvl.cpu().numpy())
        
        # save for visulsization
        save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
        if SAVE_IMAGE:
            vis_mask_save(mask_lvl[chosen_lvl], save_path)

    return chosen_iou_list, chosen_lvl_list


def lerf_localization(c_lvl, sem_map, image, clip_model, image_name, img_ann, idx=None, logger=None):
    output_path_loca = image_name / 'localization'
    output_path_loca.mkdir(exist_ok=True, parents=True)
    output_path_all_levels = image_name / 'localization_all_levels'
    output_path_all_levels.mkdir(exist_ok=True, parents=True)

    valid_map = clip_model.get_max_across(sem_map)  # 3xkx832x1264
    n_head, n_prompt, h, w = valid_map.shape
    index = 0
    # positive prompts
    acc_num = 0
    positives = list(img_ann.keys())

    for k in range(len(positives)):
        select_output = valid_map[:, k]

        # NOTE 平滑后的激活值图中找最大值点
        scale = 30
        kernel = np.ones((scale, scale)) / (scale ** 2)
        np_relev = select_output.cpu().numpy()
        avg_filtered = cv2.filter2D(np_relev.transpose(1, 2, 0), -1, kernel)

        score_lvl = np.zeros((n_head,))
        coord_lvl = []
        for i in range(n_head):  # level
            score = avg_filtered[..., i].max()
            coord = np.nonzero(avg_filtered[..., i] == score)
            score_lvl[i] = score
            coord_lvl.append(np.asarray(coord).transpose(1, 0)[..., ::-1])
        # 按分数选一个level
        # choosen_level = np.argmax(score_lvl)
        choosen_level = c_lvl[index]
        index += 1
        coord_final = coord_lvl[choosen_level]

        coord_final_list = [coord_lvl[0], coord_lvl[1], coord_lvl[2]]

        suffix = f"{clip_model.positives[k]}{idx:0>5}"

        # 只要定位成功一个就行
        for box in img_ann[positives[k]]['bboxes'].reshape(-1, 4):  # 所有box
            flag = 0
            x1, y1, x2, y2 = box
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            for cord_list in coord_final:
                if (cord_list[0] >= x_min and cord_list[0] <= x_max and
                        cord_list[1] >= y_min and cord_list[1] <= y_max):
                    acc_num += 1
                    flag = 1
                    break
            if flag != 0:
                break  # 不必再遍历其他box
        if flag == 0 and logger is not None:
            logger.info(f"[{suffix}] we choose lv {choosen_level + 1} but fail")

        for i in [0, 1, 2]:
            box_index = 0
            for box in img_ann[positives[k]]['bboxes'].reshape(-1, 4):
                flag = 0
                x1, y1, x2, y2 = box
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                for cord_list in coord_final_list[i]:
                    if (cord_list[0] >= x_min and cord_list[0] <= x_max and  # x在框的范围内
                            cord_list[1] >= y_min and cord_list[1] <= y_max):  # y在框的范围内
                        flag = 1
                        if logger is not None:
                            logger.info(
                                f"[{suffix}][level success {i + 1}][box {box_index + 1}] "
                                f"[{cord_list[0]},{cord_list[1]}] in [{x_min},{x_max},{y_min},{y_max}]")
                        break
                    else:
                        if logger is not None:
                            logger.info(
                                f"[{suffix}][level fail {i + 1}][box {box_index + 1}] "
                                f"[{cord_list[0]},{cord_list[1]}] not in [{x_min},{x_max},{y_min},{y_max}]")
                if flag != 0:
                    break
                box_index += 1

        # NOTE 将平均后的结果与原结果相加，抑制噪声并保持激活边界清晰
        avg_filtered_tensor = torch.from_numpy(avg_filtered[..., choosen_level]).unsqueeze(-1).to(select_output.device)
        torch_relev = 0.5 * (avg_filtered_tensor + select_output[choosen_level].unsqueeze(-1))
        p_i = torch.clip(torch_relev - 0.5, 0, 1)
        valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
        mask = (torch_relev < 0.5).squeeze()
        valid_composited[mask, :] = image[mask, :] * 0.3

        # 保存选中level的图
        save_path = output_path_loca / f"{positives[k]}.png"
        show_result(valid_composited.cpu().numpy(), coord_final,
                    img_ann[positives[k]]['bboxes'], save_path)

        # 保存所有level的图
        for i in range(n_head):  # n_head=3
            avg_map = torch.from_numpy(avg_filtered[..., i]).unsqueeze(-1).to(select_output.device)
            composed = 0.5 * (avg_map + select_output[i].unsqueeze(-1))
            p_i_lvl = torch.clip(composed - 0.5, 0, 1)

            color_map = colormaps.apply_colormap(p_i_lvl / (p_i_lvl.max() + 1e-6), colormaps.ColormapOptions("turbo"))
            mask_lvl = (composed < 0.5).squeeze()
            color_map[mask_lvl, :] = image[mask_lvl, :] * 0.3

            save_path_lvl = output_path_all_levels / f"{positives[k]}_level_{i + 1}.png"
            show_result(color_map.cpu().numpy(), coord_lvl[i], img_ann[positives[k]]['bboxes'], save_path_lvl)

    return acc_num



def evaluate(feat_dir, output_path, ae_ckpt_path, json_folder, mask_thresh, encoder_hidden_dims, decoder_hidden_dims, logger):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )

    gt_ann, image_shape, image_paths = eval_gt_lerfdata(Path(json_folder), Path(output_path))
    eval_index_list = [int(idx) for idx in list(gt_ann.keys())]
    compressed_sem_feats = np.zeros((len(feat_dir), len(eval_index_list), *image_shape, 3), dtype=np.float32)
    for i in range(len(feat_dir)):
        logger.info(f"Loading features from {feat_dir[i]}")
        feat_paths_lvl = sorted(glob.glob(os.path.join(feat_dir[i], '*.npy')),
                               key=lambda file_name: int(os.path.basename(file_name).split(".npy")[0]))
        for j, idx in enumerate(eval_index_list):
            logger.info(f"length of feat_paths_lvl: {len(feat_paths_lvl)}")
            logger.info(compressed_sem_feats[i][j].shape)
            logger.info(np.load(feat_paths_lvl[idx]).shape)
            compressed_sem_feats[i][j] = np.load(feat_paths_lvl[idx])

    # instantiate autoencoder and openclip
    clip_model = OpenCLIPNetwork(device)
    checkpoint = torch.load(ae_ckpt_path, map_location=device)
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    chosen_iou_all, chosen_lvl_list = [], []
    acc_num = 0
    for j, idx in enumerate(tqdm(eval_index_list)):
        image_name = Path(output_path) / f'{idx+1:0>5}'
        scene_name = os.path.basename(output_path)
        image_name.mkdir(exist_ok=True, parents=True)
        
        sem_feat = compressed_sem_feats[:, j, ...]
        sem_feat = torch.from_numpy(sem_feat).float().to(device)
        rgb_img = cv2.imread(image_paths[j])[..., ::-1]
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img).to(device)

        with torch.no_grad():
            lvl, h, w, _ = sem_feat.shape
            restored_feat = model.decode(sem_feat.flatten(0, 2))
            restored_feat = restored_feat.view(lvl, h, w, -1)           # 3x832x1264x512
        
        img_ann = gt_ann[f'{idx}']
        clip_model.set_positives(list(img_ann.keys()))
        
        c_iou_list, c_lvl = activate_stream(restored_feat, rgb_img, clip_model, image_name, img_ann,
                                            thresh=mask_thresh, colormap_options=colormap_options, idx = idx + 1, scene_name = scene_name, logger = logger)
        chosen_iou_all.extend(c_iou_list)
        chosen_lvl_list.extend(c_lvl)

        acc_num_img = lerf_localization(c_lvl, restored_feat, rgb_img, clip_model, image_name, img_ann, idx + 1, logger)
        acc_num += acc_num_img



    # localization acc
    total_bboxes = 0
    for img_ann in gt_ann.values():
        total_bboxes += len(list(img_ann.keys()))
    acc = acc_num / total_bboxes
    logger.info(f"We total have {total_bboxes} bboxes, and got {acc_num} correct")
    # # iou
    mean_iou_chosen = sum(chosen_iou_all) / len(chosen_iou_all)
    logger.info(f"trunc thresh: {mask_thresh}")
    logger.info(f"iou chosen: {mean_iou_chosen:.4f}")
    logger.info(f"chosen_lvl: \n{chosen_lvl_list}")
    logger.info(f"localization accuracy: {acc:.4f}")


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
    
    parser = ArgumentParser(description="prompt any label")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument('--feat_dir', type=str, default=None)
    parser.add_argument("--ae_ckpt_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--json_folder", type=str, default=None)
    parser.add_argument("--mask_thresh", type=float, default=0.4)
    parser.add_argument('--encoder_dims',
                        nargs = '+',
                        type=int,
                        default=[256, 128, 64, 32, 3],
                        )
    parser.add_argument('--decoder_dims',
                        nargs = '+',
                        type=int,
                        default=[16, 32, 64, 128, 256, 256, 512],
                        )
    args = parser.parse_args()

    # NOTE config setting
    dataset_name = args.dataset_name
    mask_thresh = args.mask_thresh
    feat_dir = [os.path.join(args.feat_dir, dataset_name, "train/ours_None", f"renders_npy{i}") for i in [1,2,3]]
    output_path = os.path.join(args.output_dir, dataset_name)
    ae_ckpt_path = os.path.join(args.ae_ckpt_dir, dataset_name, "best_ckpt.pth")
    json_folder = os.path.join(args.json_folder, dataset_name)

    log_path = os.path.join(output_path, "logs")
    logger = get_logger(dataset_name, log_path)
    logger.info("lerf eval started at {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    evaluate(feat_dir, output_path, ae_ckpt_path, json_folder, mask_thresh, args.encoder_dims, args.decoder_dims, logger)
