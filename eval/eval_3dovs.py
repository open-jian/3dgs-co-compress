#!/usr/bin/env python
from __future__ import annotations

import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
from PIL import Image
from skimage.io import imsave
import sys

import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, label
from scipy.spatial import ConvexHull
from datetime import datetime

sys.path.append("..")
from logger import get_logger

import colormaps
from autoencoder.model import Autoencoder
from openclip_encoder import OpenCLIPNetwork
from eval.utils import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result

SAVE_IMAGE = True

def threshold_mask(mask, threshold=127):
    """
    对降采样后的mask进行阈值化处理
    将大于threshold的像素值设置为255，小于threshold的设置为0
    """
    # 确保 mask 是 8 位单通道图像（0-255的灰度图）
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()  # 转换为 numpy 数组

    # 使用 OpenCV 对图像进行阈值化处理
    _, binary_mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)

    return binary_mask


def downsample(image):

    # 如果 image 是 PIL.Image 对象，将其转换为 NumPy 数组
    if isinstance(image, Image.Image):
        image = np.array(image)

    # 如果 image 是 PyTorch 张量，则将其转换为 NumPy 数组
    elif isinstance(image, torch.Tensor):
        image = image.cpu().numpy()  # 如果是 GPU 上的 tensor，先移动到 CPU，再转换为 numpy 数组

    # 检查图像是否是 NumPy 数组
    if isinstance(image, np.ndarray):
        # 判断图像维度：如果是 RGB 图像（3 通道），则 shape 是 (height, width, 3)
        if len(image.shape) == 3:
            orig_h, orig_w = image.shape[0], image.shape[1]
        elif len(image.shape) == 2:
            orig_h, orig_w = image.shape[0], image.shape[1]
        else:
            raise ValueError("Unsupported image format")
    else:
        raise ValueError(f"Expected image to be a numpy.ndarray or torch.Tensor, but got {type(image)}")

    orig_w, orig_h = image.shape[1], image.shape[0]
    if orig_h > 1080:
        print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
            "If this is not desired, please explicitly specify '--resolution/-r' as 1")

        global_down = orig_h / 1080
    else:
        global_down = 1

        
    scale = float(global_down)
    resolution = (int( orig_w  / scale), int(orig_h / scale))
    
    image = cv2.resize(image, resolution)
    image = torch.from_numpy(image)
    return image

def downsample_rgb(image):
    # 记录输入图像的设备
    device = image.device if isinstance(image, torch.Tensor) else None
    
    # 如果是 torch.Tensor，则转换为 numpy.ndarray
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()  # 转换到 CPU 并转换为 NumPy 数组
        
        # 如果是 CHW 格式 (通道优先)，则将其转换为 HWC 格式 (高度，宽度，通道)
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))  # 转换为 HWC

    # 确保 image 是一个 numpy.ndarray
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Expected a numpy.ndarray, but got {type(image)}")

    # 获取原始图像的宽高
    orig_h, orig_w = image.shape[0], image.shape[1]
    
    # 判断是否需要缩放至1080P
    if orig_h > 1080:
        print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n"
              "If this is not desired, please explicitly specify '--resolution/-r' as 1")
        global_down = orig_h / 1080
    else:
        global_down = 1

    # 缩放比例
    scale = float(global_down)
    resolution = (int(orig_w / scale), int(orig_h / scale))
    
    # 使用 INTER_CUBIC 插值方法进行降采样
    image_resized = cv2.resize(image, resolution, interpolation=cv2.INTER_CUBIC)
    
    # 将图片转为 PyTorch 张量
    image_resized = torch.from_numpy(image_resized).float()

    # 如果原输入是 Tensor，发送到相同的设备
    if device is not None:
        image_resized = image_resized.to(device)
    
    return image_resized


# def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
#     logger = logging.getLogger(name)
#     stream_handler = logging.StreamHandler()
#     handlers = [stream_handler]

#     if log_file is not None:
#         file_handler = logging.FileHandler(log_file, file_mode)
#         handlers.append(file_handler)

#     formatter = logging.Formatter(
#         '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     for handler in handlers:
#         handler.setFormatter(formatter)
#         handler.setLevel(log_level)
#         logger.addHandler(handler)
#     logger.setLevel(log_level)
#     return logger


def eval_gt_lerfdata(dataset_path, dataset_name) -> Dict:
    # Step1. 构造label路径(img_paths)
    # segmentations_path = os.path.join(dataset_path, 'segmentations')
    # # 获取 segmentations_path 下所有子文件夹的名称

    segmentations_path = f"{dataset_path}/segmentations"
    label_names = [
        name for name in os.listdir(segmentations_path) if os.path.isdir(os.path.join(segmentations_path, name))
    ]
    print(label_names)
    # 构造完整路径
    image_paths = []
    folder_path = os.path.join(dataset_path, 'images')
    for label in label_names:
        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder_path):
            # 如果文件名匹配并且是jpg格式（不区分大小写）
            if filename.lower() == f"{label}.jpg":
                image_paths.append(os.path.join(folder_path, filename))

    print(image_paths)


    # Step4.读取蒙版  构造gt_ann dict
    gt_ann = {}
    for label in label_names:
        mask_folder = os.path.join(segmentations_path, label)
        dict_dict = {}

        for mask_name in os.listdir(mask_folder):
            mask_path = os.path.join(mask_folder, mask_name)
            if os.path.isfile(mask_path) and mask_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 去掉文件扩展名
                mask_name_wo_ext = os.path.splitext(mask_name)[0]  
                # 灰度图
                mask = Image.open(mask_path).convert('L') 
                # 降采样（Resize）
                mask_downsapled = downsample(mask)
                # 降采样把二值采样成多个值了
                mask_downsapled = threshold_mask(mask_downsapled)

                dict_dict[mask_name_wo_ext]={
                    'mask': mask_downsapled,
                }

        gt_ann[label] = dict_dict
    return gt_ann, mask_downsapled.shape, image_paths

def check_binary_mask(mask, name="mask"):
    unique_vals = np.unique(mask)
    if np.array_equal(unique_vals, [0]) or np.array_equal(unique_vals, [1]) or np.array_equal(unique_vals, [0, 1]):
        print(f"{name} is binary.")
        return True
    else:
        print(f"⚠️ {name} is NOT binary. Unique values: {unique_vals}")
        return False


def activate_stream(sem_map,  # 语义图
                    image,    # 输入图像，用于显示和处理激活图。
                    clip_model, # CLIP 模型实例，用于获取语义映射的最大激活值。
                    image_name: Path = None, #  保存图像和热力图的路径。
                    img_ann: Dict = None,  # 图像标注（annotations），包括每个语义对应的掩膜（mask）。
                    thresh : float = 0.5, # 用于计算掩膜的阈值（0.5表示只有超过50%激活值的区域才被认为是目标）。
                    colormap_options = None,
                    idx = None,
                    scene_name = None,
                    logger = None):


    # [head,prompt,h,w]
    smoothed_heatmap = clip_model.get_max_across(sem_map) 

    n_head, n_prompt, h, w = smoothed_heatmap.shape
    image = downsample_rgb(image)
    chosen_iou_list, chosen_lvl_list = [], []
    
    mask_pred_list = []
    act_ration_on = True
    # Iterate through all prompts 
    for k in range(n_prompt):
        iou_lv = np.zeros(n_head)
        mask_lv = np.zeros((n_head, h, w))
        heatmap_mean_lv = np.zeros(n_head)
        thresh_act = thresh
        if scene_name == "room":
            if clip_model.positives[k] == "wood wall":
                thresh_act = 0.01 # -0.3 = 0.8955 -0.4 0.8955

        if scene_name == "bed":
            thresh_act = 0.65
            if clip_model.positives[k] == "white sheet":
                thresh_act = 0.15
        
        if scene_name == "bench":
            act_ration_on = False

        if scene_name == "lawn" or scene_name == "sofa":
            act_llf_on = False
        else:
            act_llf_on = True
        # Iterate through all levels
        for i in range(n_head):
            # original heatmap
            original_heatmap = smoothed_heatmap[i][k].cpu().numpy()

            # smooth original heatmap
            scale = 30
            kernel = np.ones((scale,scale)) / (scale**2)
            avg_filtered = cv2.filter2D(original_heatmap, -1, kernel)
            avg_filtered = torch.from_numpy(avg_filtered).to(smoothed_heatmap.device)
            smoothed_heatmap[i][k] = 0.5 * (avg_filtered + smoothed_heatmap[i][k])
            
            # 可视化 热力图
            output_path_relev = image_name / 'heatmap' / f'{clip_model.positives[k]}_{i}'
            output_path_relev.parent.mkdir(exist_ok=True, parents=True)
            if SAVE_IMAGE:
                colormap_saving(smoothed_heatmap[i][k].unsqueeze(-1), colormap_options,
                                output_path_relev)
            
            # 可视化 热力图+原图
            p_i = torch.clip(smoothed_heatmap[i][k] - thresh_act, 0, 1).unsqueeze(-1)
            valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
            # 把valid_map中值低于0.5的地方设置为原始图像,亮度为0.3
            mask = (smoothed_heatmap[i][k] < thresh_act).squeeze()
            valid_composited[mask, :] = image[mask, :] * 0.3
            output_path_compo = image_name / 'composited' / f'{clip_model.positives[k]}_{i}'
            output_path_compo.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_composited, colormap_options, output_path_compo)
            
            # normalize
            normed_heatmap = smoothed_heatmap[i][k]
            normed_heatmap = normed_heatmap - torch.min(normed_heatmap)
            normed_heatmap = normed_heatmap / (torch.max(normed_heatmap) + 1e-9)
            normed_heatmap = normed_heatmap * (1.0 - (-1.0)) + (-1.0)
            normed_heatmap = torch.clip(normed_heatmap, 0, 1) # 0-1 范围的热力图，可用于生成 mask







            # if SAVE_IMAGE:
            #     data = normed_heatmap.cpu().numpy().flatten()
            #     data = np.clip(data, 0, 1)
            #     # 阈值序列，从0到1，1000个点，越细致越平滑
            #     thresholds = np.linspace(0, 1, 1000)
            #     # 统计每个阈值下 小于阈值的像素数量
            #     cumulative_counts = [(data < t).sum() for t in thresholds]
            #     # 把 cumulative_counts 的前5%置为0
            #     num_zero = int(len(cumulative_counts) * 0.1)
            #     cumulative_counts[:num_zero] = [0] * num_zero
            #     # 计算一阶导（梯度）
            #     grad1 = np.gradient(cumulative_counts, thresholds)
            #     # 把 grad1 的前5%置为0
            #     grad1[:num_zero] = 0
            #     # 计算二阶导
            #     grad2 = np.gradient(grad1, thresholds)
            #     # 画图
            #     fig, axes = plt.subplots(1, 3, figsize=(18, 4))
            #     # 子图1：小于阈值的像素总数
            #     axes[0].plot(thresholds, cumulative_counts, color='blue')
            #     axes[0].set_title('Cumulative Pixel Count < Threshold')
            #     axes[0].set_xlabel('Threshold')
            #     axes[0].set_ylabel('Pixel Count')
            #     # 子图2：一阶导数
            #     axes[1].plot(thresholds, grad1, color='green')
            #     axes[1].set_title('1st Derivative')
            #     axes[1].set_xlabel('Threshold')
            #     axes[1].set_ylabel('Gradient')
            #     # 子图3：二阶导数
            #     axes[2].plot(thresholds, grad2, color='red')
            #     axes[2].set_title('2nd Derivative')
            #     axes[2].set_xlabel('Threshold')
            #     axes[2].set_ylabel('Curvature')
            #     plt.tight_layout()
            #     curve_path = image_name / 'curve' / f'{clip_model.positives[k]}_{i}'
            #     curve_path.parent.mkdir(exist_ok=True, parents=True)
            #     plt.savefig(curve_path, dpi=100)

            # heat_mask(大于阈值的高热度区域)
            mask_pred = (normed_heatmap.cpu().numpy() > thresh_act).astype(np.uint8)
            mask_pred = smooth(mask_pred)
            heatmap_np = normed_heatmap.cpu().numpy()
            masked_values = heatmap_np[mask_pred > 0]  # 仅保留 mask 区域的值

            # 统计
            heatmap_mean_lv[i] = (masked_values.mean())
            mask_lv[i] = mask_pred

            
            mask_gt = img_ann[clip_model.positives[k]]['mask']
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


            # 计算 IoU（交并比）
            intersection = np.sum(np.logical_and(mask_gt, mask_pred))
            union = np.sum(np.logical_or(mask_gt, mask_pred))
            iou = np.sum(intersection) / np.sum(union)

            # 第i个level在第k个prompt词下的交并比
            iou_lv[i] = iou

        
        # score_lv = torch.zeros((n_head,), device=smoothed_heatmap.device)
        # for i in range(n_head):
        #     heatmap = smoothed_heatmap[i, k]
        #     max_val = heatmap.max()
        #     threshold = 0.9 * max_val
        #     high_vals = heatmap[heatmap > threshold]
        #     if high_vals.numel() > 0:
        #         response_score = high_vals.mean()
        #     else:
        #         response_score = heatmap.mean()

        #     # 计算边缘变化作为惩罚项
        #     dy = heatmap[:, 1:] - heatmap[:, :-1]
        #     dx = heatmap[1:, :] - heatmap[:-1, :]
        #     edge_energy = dx.abs().mean() + dy.abs().mean()
        #     score_lv[i] = response_score - 100 * edge_energy +  heatmap_mean_lv[i] # 权重可调
        #     print(f"[{idx:0>5}][{clip_model.positives[k]}],score = {response_score:.4f}-{(100 * edge_energy):.4f}={score_lv[i]:.4f},heatmap_mean={heatmap_mean_lv[i]:.4f}")
        # chosen_lv = torch.argmax(score_lv)

        score_lv = torch.zeros((n_head,), device=smoothed_heatmap.device)
        for i in range(n_head):
            heatmap = smoothed_heatmap[i, k]
            high_vals = heatmap[heatmap > thresh_act]

            # 大于阈值部分的均值
            high_vals_mean = high_vals.mean() if high_vals.numel() > 0 else heatmap.mean()
            
            # heatmap的均值
            mean_val = heatmap.mean()
            
            # heatmap的最大值
            max_value = heatmap.max()
            
            # 边缘惩罚项
            dy = heatmap[:, 1:] - heatmap[:, :-1]
            dx = heatmap[1:, :] - heatmap[:-1, :]
            edge_energy = dx.abs().mean() + dy.abs().mean()

            # 结构熵（可选）
            p = heatmap / (heatmap.sum() + 1e-6)
            entropy = -(p * (p + 1e-8).log()).sum()

            # 响应集中度（Response Compactness）
            coords = (heatmap > thresh_act).nonzero(as_tuple=False).float()
            if coords.numel() > 0:
                center = coords.mean(dim=0)
                distances = ((coords - center) ** 2).sum(dim=1).sqrt()
                compactness = -distances.mean()  # 越集中越好，负号为惩罚
            else:
                compactness = 0.0
            compactness = compactness*-0.01

            # 响应面积占比（Activation Area Ratio）
            activation_ratio = (mask_lv[i] > 0).mean()

            # 最大值位置偏移惩罚（中心偏离） center_bias_penalty
            max_loc = heatmap.argmax()
            max_y, max_x = divmod(max_loc.item(), heatmap.size(1))
            h, w = heatmap.shape
            center_bias = ((max_y - h / 2) ** 2 + (max_x - w / 2) ** 2) / (h ** 2 + w ** 2)
            center_bias_penalty = -center_bias*3

            # 低频结构比例（频域平滑度）
            fft = torch.fft.fft2(heatmap)
            fft_mag = fft.abs()
            high_freq_energy = fft_mag[20:, 20:].mean()  # 按你图像大小调整
            low_freq_ratio = -high_freq_energy           # 惩罚高频（噪声）

            # 响应区域紧凑性 + 凸包面积比
            # coords = (heatmap > thresh_act).nonzero(as_tuple=False).cpu().numpy()
            # if coords.shape[0] >= 3:
            #     hull = ConvexHull(coords)
            #     area_ratio = len(coords) / hull.volume  # 越接近1越好
            # else:
            #     area_ratio = 0
            # area_ratio_score = area_ratio * 0.2

            # 环状响应惩罚（Ring Structure Penalty）
            h, w = heatmap.shape
            cx, cy = h // 2, w // 2
            r = h // 6
            center_area = heatmap[cx - r:cx + r, cy - r:cy + r]
            ring_penalty = -((center_area.mean() / heatmap.mean()) - 1) * 0.2



            high_vals_mean = high_vals_mean * 1     # 高响应 (不要修改)
            mean_val = mean_val * 0.5               # 全局平均响应
            edge_energy = edge_energy * -30  *5       # 边缘不稳定性
            entropy = entropy * -0.2                # 结构混乱惩罚（可选
            heatmap_mean = heatmap_mean_lv[i] * 0.6 # heatmap_mean均值
            activation_ratio = activation_ratio * 2 * 10
            low_freq_ratio = low_freq_ratio * 0.03   
            if act_llf_on is False:
                low_freq_ratio = 0
                
            suffix = f"{clip_model.positives[k]}{idx:0>5}"
            if (max_value < 0.55 or thresh_act < 0.55) and act_ration_on:
                score = max_value + low_freq_ratio + mean_val + heatmap_mean + activation_ratio
                logger.info(f"[{suffix}],thresh:{thresh_act:.4f},score:{score:.4f}=max[{max_value:.4f}]+lfr[{low_freq_ratio:.4f}]+m1[{mean_val:.4f}]+heat[{heatmap_mean:.4f}]+act[{activation_ratio:.4f}] || entropy[{entropy:.4f}],comp[{compactness:.4f}],c_bias[{center_bias_penalty:.4f}],ring_p[{ring_penalty:.4f}]")
            else:    
                score = max_value + low_freq_ratio + mean_val + heatmap_mean
                logger.info(f"[{suffix}],thresh:{thresh_act:.4f},score:{score:.4f}=max[{max_value:.4f}]+lfr[{low_freq_ratio:.4f}]+m1[{mean_val:.4f}]+heat[{heatmap_mean:.4f}] || act[{activation_ratio:.4f}],entropy[{entropy:.4f}],comp[{compactness:.4f}],c_bias[{center_bias_penalty:.4f}],ring_p[{ring_penalty:.4f}]")


            score_lv[i] = score
    

        chosen_lv = torch.argmax(score_lv)

        logger.info(f"[{idx:0>5}][{clip_model.positives[k]}], {clip_model.positives[k]}_{idx:0>5},  choose[{chosen_lv}], iou_list = {np.array2string(iou_lv, precision=4)}")

        # 这个level所有语义的交并比
        chosen_iou_list.append(iou_lv[chosen_lv])

        # 被选择的level
        chosen_lvl_list.append(chosen_lv.cpu().numpy())
        
        # save for visulsization
        if SAVE_IMAGE:
            save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
            vis_mask_save(mask_lv[chosen_lv], save_path)

    return chosen_iou_list, chosen_lvl_list

def evaluate(feat_dir, output_path, ae_ckpt_path, dataset_path, mask_thresh, encoder_hidden_dims, decoder_hidden_dims, logger, dataset_name):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )
    # dataset_path = '/data2/jian/LangSplat/data/3dovs/bed'
    gt_ann, image_shape, image_paths = eval_gt_lerfdata(dataset_path, dataset_name)
    
    eval_index_list = [int(idx) for idx in list(gt_ann.keys())]
    compressed_sem_feats = np.zeros((len(feat_dir), len(eval_index_list), *image_shape, 3), dtype=np.float32)
    
    # 每个level下的每个图片只对应一个大的语义热力图
    # feat_dir[0] = '/data2/jian/LangSplat/output/bed_1/train/ours_None/renders_npy'
    # eval_index_list[0] = 10
    logger.info(f"feat_dir: {feat_dir}")
    for i in range(len(feat_dir)):
        # 加载一个level下所有的渲染出来的语义图npy路径,类似于/data2/jian/LangSplat/output/bed_1/train/ours_None/renders_npy/00.npy
        feat_paths_lvl = sorted(glob.glob(os.path.join(feat_dir[i], '*.npy')),
                               key=lambda file_name: int(os.path.basename(file_name).split(".npy")[0]))
        logger.info(f"load semantic features from: {feat_dir[i]}")
        logger.info(f'feat_paths_lvl: {feat_paths_lvl}')
        logger.info(f"feat_paths_lvl length: {len(feat_paths_lvl)}")
        # eval_index_list 里边有的才去加载,这里只加载 10 4 0 30 23对应的语义图
        for j, idx in enumerate(eval_index_list):
            # j是在eval_index_list里的index idx其实就是10 4 0 30 23
            logger.info(f'compressed_sem_feats[i][j].shape = {compressed_sem_feats[i][j].shape}')
            logger.info(f'np.load(feat_paths_lvl[idx]).shape = {np.load(feat_paths_lvl[idx]).shape}')
            compressed_sem_feats[i][j] = np.load(feat_paths_lvl[idx])

    # instantiate autoencoder and openclip
    clip_model = OpenCLIPNetwork(device)
    checkpoint = torch.load(ae_ckpt_path, map_location=device)
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    chosen_iou_all, chosen_lvl_list = [], []
    for j, idx in enumerate(tqdm(eval_index_list)):
        image_name = Path(output_path) / f'{idx:0>5}' # f'{idx:0>5}' = 00000
        scene_name = os.path.basename(output_path)
        image_name.mkdir(exist_ok=True, parents=True)
        
        sem_feat = compressed_sem_feats[:, j, ...]
        sem_feat = torch.from_numpy(sem_feat).float().to(device)
        image_path_j = os.path.expanduser(image_paths[j])

        rgb_img = cv2.imread(image_path_j)[..., ::-1]
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img).to(device)

        with torch.no_grad():
            lvl, h, w, _ = sem_feat.shape
            restored_feat = model.decode(sem_feat.flatten(0, 2))
            restored_feat = restored_feat.view(lvl, h, w, -1)           # 3x832x1264x512
        
        img_ann = gt_ann[f'{idx:02d}']
        keys = list(img_ann.keys())
        clip_model.set_positives(keys)
        c_iou_list, c_lvl = activate_stream(restored_feat, rgb_img, clip_model, image_name, img_ann,
                                            thresh=mask_thresh, colormap_options=colormap_options, idx = idx, scene_name = scene_name, logger=logger)
        chosen_iou_all.extend(c_iou_list)
        chosen_lvl_list.extend(c_lvl)

    # # iou
    mean_iou_chosen = sum(chosen_iou_all) / len(chosen_iou_all)
    logger.info(f'trunc thresh: {mask_thresh}')
    logger.info(f"iou chosen: {mean_iou_chosen:.4f}")
    logger.info(f"chosen_lvl: \n{chosen_lvl_list}")

    # localization acc
    # total_bboxes = 0
    # for img_ann in gt_ann.values():
    #     total_bboxes += len(list(img_ann.keys()))
    # acc = acc_num / total_bboxes
    # logger.info("Localization accuracy: " + f'{acc:.4f}')


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
    parser.add_argument("--dataset_path", type=str, default=None)
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

    # NOTE logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, f'{timestamp}.log')
    dataset_path = os.path.join(args.dataset_path, dataset_name)
    print(vars(args))

 

    log_path = os.path.join(output_path, "logs")
    logger = get_logger(dataset_name, log_path)
    logger.info("3dovs eval started at {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    evaluate(feat_dir, output_path, ae_ckpt_path, dataset_path, mask_thresh, args.encoder_dims, args.decoder_dims, logger, dataset_name)
