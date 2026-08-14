import glob
import os
import numpy as np
import torch
import argparse
import shutil
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torchvision.transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from autoencoder.dataset import Autoencoder_dataset, seg_dataset
from autoencoder.model import Autoencoder
from preprocess import OpenCLIPNetwork
from dataclasses import dataclass, field
from typing import Tuple, Type
from PIL import Image
from torchmetrics import JaccardIndex
from torch.utils.data import Dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='bed')
    parser.add_argument('--gt_path', type=str, default=None)
    parser.add_argument('--ae_checkpoint', type=str, default=None)
    parser.add_argument('--encoder_dims',nargs='+',type=int,default=[256, 128, 64, 32, 3],)
    parser.add_argument('--decoder_dims',nargs='+',type=int,default=[16, 32, 64, 128, 256, 256, 512],)
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    code_root = os.path.dirname(repo_root)
    output_root = os.path.abspath(
        os.environ.get("OUTPUT_ROOT", os.path.join(os.path.dirname(code_root), "Output"))
    )
    data_root = os.path.abspath(
        os.environ.get("DATA_ROOT", os.path.join(os.path.dirname(code_root), "Data"))
    )
    if args.output_path is None:
        args.output_path = os.path.join(output_root, "colasplat", "semantic", "3dovs")
    if args.gt_path is None:
        args.gt_path = os.path.join(data_root, "3dovs", args.dataset_name, "segmentations")

    dataset_name = args.dataset_name
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims
    # ckpt_path = f"pretrained_model/autoencoder/sofa/best_ckpt.pth"
    ckpt_path = args.ae_checkpoint or os.path.join(
        output_root, "colasplat", "autoencoder", "ckpt", dataset_name, "best_ckpt.pth"
    )

    # data_dir = f"{args.output_path}/train/ours_None/renders_npy"
    data_dir = f'{args.output_path}/{args.dataset_name}'
    output_dir = f"{args.output_path}/seg"


    test_views = glob.glob(f"{args.gt_path}/*")
    print("This is  ckpt_path:{}".format(ckpt_path))
    test_views_str = sorted([item.split('/')[-1] for item in test_views if os.path.isdir(item)])
    # print(test_views)

    checkpoint = torch.load(ckpt_path)
    train_dataset = seg_dataset(data_dir, test_views)

    test_loader = DataLoader(
        dataset=train_dataset,
        batch_size=256*6075,
        shuffle=False,
        num_workers=16,
        drop_last=False
    )
    jaccard = JaccardIndex(task="binary", num_classes=2)
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

    model.load_state_dict(checkpoint)
    model.eval()


@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ('Pikachu',
                             'a stack of UNO cards',
                             'a red Nintendo Switch joy-con controller',
                             'Gundam',
                             'Xbox wireless controller',
                             'grey sofa')

CLIP_model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
relevancy_masks = []
relevancys = []
gt_masks = []
for idx, feature in tqdm(enumerate(test_loader)):
    data = feature.to("cuda:0")
    with torch.no_grad():
        outputs = model.decode(data)
    # if idx == 0:
    features = outputs
    # else:
    #     features = np.concatenate([features, outputs], axis=0)
    # print(features.shape)
    for positive_id in range(len(OpenCLIPNetworkConfig.positives)):
        relevancy = CLIP_model.get_relevancy(features, positive_id).view(train_dataset.data_dic[idx % len(test_views_str)][0], train_dataset.data_dic[idx % len(test_views_str)][1], 2).permute(2,0,1)
        # print(relevancy)
        gt_mask_path = f'{args.gt_path}/{test_views_str[idx % len(test_views_str)]}/{OpenCLIPNetworkConfig.positives[positive_id]}.png'
        gt_mask = torchvision.transforms.ToTensor()(Image.open(gt_mask_path)).cuda()
        gt_mask = 0.2989 * gt_mask[0] + 0.5870 * gt_mask[1] + 0.1140 * gt_mask[2]
        # relevancy_mask = torch.argmin(relevancy, dim=0).float()
        relevancy_mask = torch.zeros_like(relevancy[0])
        relevancy_mask[relevancy[0] > 0.6] = 1
        # relevancy_mask = torch.nn.functional.interpolate(relevancy_mask.unsqueeze(0).unsqueeze(0), gt_mask.shape, mode='nearest')
        relevancy_masks.append(relevancy_mask.cpu())
        relevancys.append(relevancy[0].cpu())
        gt_mask[gt_mask>0.4] = 1

        gt_masks.append(gt_mask.cpu())
        relevancy_img = torchvision.transforms.ToPILImage()(relevancy_mask)
        render_dir = os.path.join(output_dir, "scales", str(idx // 5), test_views_str[idx % len(test_views_str)])
        os.makedirs(render_dir, exist_ok=True)
        relevancy_img.save(os.path.join(render_dir, f"{OpenCLIPNetworkConfig.positives[positive_id]}.png"))
        # print(relevancy.shape)
        # print(gt_mask.shape)
        # miou = jaccard(relevancy.squeeze(0), gt_mask.unsqueeze(0))
        # MIoU.append(miou.data)
        # print(miou)
# print(sum(MIoU)/len(MIoU))

    # break
num_imgs = len(test_views_str)
MIoU = []
ACC = []
# print(len(relevancys))
for i in tqdm(range(num_imgs)):
    for positive_id in range(len(OpenCLIPNetworkConfig.positives)):
        relevancy_1 = relevancys[i*len(OpenCLIPNetworkConfig.positives) + positive_id]
        relevancy_2 = relevancys[i*len(OpenCLIPNetworkConfig.positives) + positive_id + len(OpenCLIPNetworkConfig.positives) * num_imgs]
        relevancy_3 = relevancys[i*len(OpenCLIPNetworkConfig.positives) + positive_id + len(OpenCLIPNetworkConfig.positives) * num_imgs * 2]


        relevancy_mask_1 = relevancy_masks[i*len(OpenCLIPNetworkConfig.positives) + positive_id]
        relevancy_mask_2 = relevancy_masks[i*len(OpenCLIPNetworkConfig.positives) + positive_id + len(OpenCLIPNetworkConfig.positives) * num_imgs]
        relevancy_mask_3 = relevancy_masks[i*len(OpenCLIPNetworkConfig.positives) + positive_id + len(OpenCLIPNetworkConfig.positives) * num_imgs * 2]
        # print(relevancy_1.shape)
        # print(relevancy_mask_1.shape)
        score_1 = torch.sum(relevancy_1 * relevancy_mask_1) / torch.sum(relevancy_mask_1)
        score_2 = torch.sum(relevancy_2 * relevancy_mask_2) / torch.sum(relevancy_mask_2)
        score_3 = torch.sum(relevancy_3 * relevancy_mask_3) / torch.sum(relevancy_mask_3)

        final_mask = [relevancy_mask_1, relevancy_mask_2, relevancy_mask_3][torch.argmax(torch.tensor([score_1, score_2, score_3]))]
        # print(score_1, score_2, score_3)
        # print(torch.argmax(torch.tensor([score_1, score_2, score_3])))
        relevancy_mask = torch.nn.functional.interpolate(final_mask.unsqueeze(0).unsqueeze(0), gt_masks[i].shape,
                                                         mode='nearest')
        relevancy_img = torchvision.transforms.ToPILImage()(relevancy_mask.squeeze(0))
        render_dir = os.path.join(output_dir, "pred", test_views_str[i % len(test_views_str)])
        os.makedirs(render_dir, exist_ok=True)
        relevancy_img.save(os.path.join(render_dir, f"{OpenCLIPNetworkConfig.positives[positive_id]}.png"))

        gt_img = torchvision.transforms.ToPILImage()(gt_masks[i*len(OpenCLIPNetworkConfig.positives) + positive_id].unsqueeze(0))
        gt_dir = os.path.join(output_dir, "gt", test_views_str[i % len(test_views_str)])
        os.makedirs(gt_dir, exist_ok=True)
        gt_img.save(os.path.join(gt_dir, f"{OpenCLIPNetworkConfig.positives[positive_id]}.png"))

        miou = jaccard(relevancy_mask.squeeze(0), gt_masks[i*len(OpenCLIPNetworkConfig.positives) + positive_id].unsqueeze(0))
        acc = torch.sum(relevancy_mask.squeeze(0) == gt_masks[i*len(OpenCLIPNetworkConfig.positives) + positive_id].unsqueeze(0)) / (gt_masks[i*len(OpenCLIPNetworkConfig.positives) + positive_id].shape[0] * gt_masks[i*len(OpenCLIPNetworkConfig.positives) + positive_id].shape[1])
        ACC.append(acc)
        MIoU.append(miou)
        # print(f'{test_views_str[i % len(test_views_str)]}/{OpenCLIPNetworkConfig.positives[positive_id]}.png')
        print(miou)
        print(acc)
print(sum(MIoU) / len(MIoU))
print(sum(ACC) / len(ACC))

    # print(score_1, score_2, score_3)
