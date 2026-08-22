# LangSplatV2_Cluno

This project combines LangSplatV2's semantic representation with CoLaSplat's
joint optimization and pruning. Attribute compression uses C3DGS-style
sensitivity-aware vector quantization adapted to the ADMM loop. The upstream
LangSplatV2 checkout remains separate at `../LangSplatV2`.

## Joint CoLaSplat optimization (three scales, one model)

`train_joint.py` jointly optimizes opacity sparsity and three C3DGS-style VQ
blocks: color (DC and SH), normalized covariance, and semantic coefficients.
Euclidean assignments and sensitivity-weighted centroid updates follow the
C3DGS quantizer, while ADMM couples each projection to the rendering and
semantic objectives. It stores the small, medium, and large semantic scales as
three VQ heads in one Gaussian model. Geometry and opacity are shared, and every
physical pruning event applies the same mask to all three semantic heads.

If the released V2 `_1`, `_2`, and `_3` checkpoints are already trained, the
recommended migration path is to merge them and start the CoLaSplat stage:

```bash
bash merge_and_train_joint.sh DATASET_ROOT SCENE_NAME 0 10000 3dovs
```

The merger verifies that all three checkpoints have identical Gaussian
geometry, stacks their logits/codebooks into the scale dimension, and discards
the three incompatible per-scale optimizer states. The compression stage then
starts at iteration zero with a fresh joint optimizer.

To train from the 30k RGB checkpoint, the wrapper first learns all three heads
with frozen geometry, then resets the iteration counter and runs joint ADMM
compression:

```bash
bash train_joint.sh DATASET_ROOT SCENE_NAME 0 3dovs
```

The semantic pretraining command used by the wrapper is:

```bash
python train_joint.py \
    -s DATASET_ROOT/SCENE_NAME \
    -m ../../Output/langsplatv2_cluno/joint/3dovs/SCENE_NAME_0_pretrain \
    --start_checkpoint ../../Output/rgb_3dgs/3dovs/SCENE_NAME/chkpnt30000.pth \
    --feature_levels 1 2 3 \
    --vq_layer_num 1 --codebook_size 64 \
    --iterations 10000 --semantic_only --language_loss_coeff 1.0 \
    --semantic_loss cos --normalize --topk 4
```

The joint checkpoint is
`../../Output/langsplatv2_cluno/joint/3dovs/SCENE_NAME_0/chkpnt10000.pth` and its
self-describing multi-block quantization sidecar is
`sh_quantization_10000.pth`. The legacy filename and options such as
`--rho_sh` are retained so existing launchers keep working; they no longer mean
that only SH is quantized. The released quick rasterizer can evaluate the
single checkpoint directly, for example:

```bash
python eval_lerf.py ... --include_feature --quick_render \
    --joint_checkpoint ../../Output/langsplatv2_cluno/joint/lerf_ovs/SCENE_NAME_0/chkpnt10000.pth
```

The same `--joint_checkpoint` option is supported by `eval_3d_ovs.py` and
`eval_mip_nerf360.py`. Legacy three-directory checkpoints remain supported.
<div align="center">
  <p>
    <a href="https://li-wanhua.github.io/">Wanhua Li*</a>,
    <a href="https://github.com/ZhaoYujie2002">Yujie Zhao*</a>,
    <a href="https://minghanqin.github.io/">Minghan Qin*</a>,
    <a href="https://github.com/jimmyYliu">Yang Liu</a>,
    <a href="https://caiyuanhao1998.github.io/">Yuanhao Cai</a>,
    <a href="https://people.csail.mit.edu/ganchuang/">Chuang Gan</a>,
    <a href="https://vcg.seas.harvard.edu/people">Hanspeter Pfister</a>,
  </p>
</div>
<div align="center">
  (* indicates equal contribution)
</div>

<!--
[Wanhua Li*](https://li-wanhua.github.io/), [Yujie Zhao*](https://github.com/ZhaoYujie2002), [Minghan Qin*](https://minghanqin.github.io/),  [Yang Liu](https://github.com/jimmyYliu), [Yuanhao Cai](https://caiyuanhao1998.github.io/), [Chuang Gan](https://people.csail.mit.edu/ganchuang/), [Hanspeter Pfister](https://vcg.seas.harvard.edu/people/hanspeter-pfister)<br>(\* indicates equal contribution)<br>
-->

<div align="center">
<!--
<a href="https://jytime.github.io/data/VGGT_CVPR25.pdf" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/Paper-VGGT" alt="Paper PDF">
</a>
-->
<a href="https://arxiv.org/abs/2507.07136">
  <img src="https://img.shields.io/badge/arXiv-2507.07136-b31b1b" alt="arXiv">
</a>
<a href="https://langsplat-v2.github.io/">
  <img src="https://img.shields.io/badge/Project_Page-LangSplat_v2-green" alt="Project Page">
</a>
<a href="https://drive.google.com/drive/folders/1bHzv8e69NHseEj-3Qe0k0GqTmDYMGg-K?usp=sharing">
  <img src="https://img.shields.io/badge/Google_Drive-Model_Weights-blue" alt="Model">
</a>
</div>
<!--
| [Webpage](https://langsplat.github.io/) | [Full Paper](https://arxiv.org/pdf/2312.16084.pdf) | [Video](https://www.youtube.com/watch?v=XMlyjsei-Es) |<br>
| Preprocessed Dataset | [BaiduWangpan](https://pan.baidu.com/s/1S_cdmN9EFOlCQ3z1GZR3EA?pwd=lfea) | [GoogleDrive](https://drive.google.com/drive/folders/1Icw5AcQkY_2L_k7ddXrGCJ3z4laa4jg5?usp=sharing) |<br>
| Pre-trained Models | [BaiduWangpan](https://pan.baidu.com/s/12L83uEi5KlF9ViAZqp0B4w?pwd=dl22) | [GoogleDrive](https://drive.google.com/drive/folders/1ASFXWOwaXP_aSXV2iMDmEfILaDXQXlrE?usp=sharing) |<br>
| [Datasets](https://drive.google.com/file/d/1QF1Po5p5DwTjFHu6tnTeYs_G0egMVmHt/view?usp=sharing) |<br>
-->

<p align="center">
    <img src="assets/teaser.png" alt="Teaser" width="500" style="center" />
</p>

![Framework](assets/framework.png)

This repository contains the official authors implementation associated with the paper "High-dimensional 3D Language Gaussian Splatting with 450+ FPS". We further provide the datasets and model weights. 

## Clone the Repository

This `LangSplatV2_Cluno` branch vendors the three upstream dependency sources
under `submodules/`, including the local Segment Anything compatibility fix.
Clone this branch directly; `--recursive` is not required.

```shell
git clone --branch LangSplatV2_Cluno git@github.com:anonymous-dasdagvfa/CoLaSplat.git
```

## Setup
#### Requirements
- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions (we used VS Code)
- CUDA SDK 11 for PyTorch extensions (we used 11.8)
- C++ Compiler and CUDA SDK must be compatible
- A NVIDIA A100 GPU (we used in our experiments)

#### Environment Setup

Our default, provided install method is based on Conda package and environment management:
```shell
conda env create --file environment.yml
conda activate langsplat_v2
```

## Prepare Dataset
In the experiments section of our paper, we primarily utilized Three datasets: the LERF dataset and the 3D-OVS dataset and the Mip-NeRF360 dataset.

The LERF dataset is accessible for download from repository [LangSplat](https://github.com/minghanqin/LangSplat?tab=readme-ov-file) via the following link: [Download LERF Dataset](https://drive.google.com/file/d/1QF1Po5p5DwTjFHu6tnTeYs_G0egMVmHt/view?usp=sharing).

The 3D-OVS dataset is accessible for download from repository [3D-OVS](https://github.com/Kunhao-Liu/3D-OVS) via the following link: [Download 3D-OVS Dataset](https://drive.google.com/drive/folders/1kdV14Gu5nZX6WOPbccG7t7obP_aXkOuC?usp=sharing).

The Mip-NeRF360 dataset is accessible for download from repostory [GAGS](https://github.com/WHU-USI3DV/GAGS?tab=readme-ov-file) via the following link: [Download Mip-NeRF360 Dataset](https://drive.google.com/drive/folders/1_IbWgVgvnCy4jq9P5EcE6xS44ftcmtgq).

For your own scenes, prepare the dataset below and first train an RGB model
following [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).
In this workspace, keep the RGB checkpoint separate from the dataset at
`../../Output/rgb_3dgs/<dataset_kind>/<scene>/chkpnt30000.pth`.
```
<dataset_name>
|---images
|   |---<image 0>
|   |---<image 1>
|   |---...
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```

## QuickStart

Put the pretrained model under the workspace-level `../../Output/` layout, then use

```shell
# For the LERF dataset
# eg. bash eval_lerf.sh teatime 0 10000
bash eval_lerf.sh $scene_name $model_idx $checkpoint
```

## Pipeline

The pipeline for training LangSplat V2 and evaluation.
- **Step 1: Generate Language Feature of the Scenes.**
  ```shell
  python preprocess.py --dataset_path $dataset_path 
  ```
- **Step 2: Jointly train all three semantic scales in one Gaussian model.**
  ```shell
  bash train_joint.sh $dataset_root_path $scene_name $model_idx $dataset_kind
  ```
  `train.sh` is retained only for compatibility with legacy three-directory
  checkpoints.
- **Step 3: Eval.**

  For the LERF dataset
  ```shell
  bash eval_lerf.sh $scene_name $model_idx $checkpoint
  ```
  For the 3D-OVS dataset
  ```shell
  bash eval_3d_ovs.sh $scene_name $model_idx $checkpoint
  ```
  For the Mip-NeRF360 dataset
  ```shell
  bash eval_mip_nerf360.sh $scene_name $model_idx $checkpoint
  ```

## TODO list:
- [x] update the arxiv link
- [x] release more model weights
- [x] release more code of evaluation


<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@article{li2025langsplatv2,
      title={LangSplatV2: High-dimensional 3D Language Gaussian Splatting with 450+ FPS}, 
      author={Wanhua Li and Yujie Zhao and Minghan Qin and Yang Liu and Yuanhao Cai and Chuang Gan and Hanspeter Pfister},
      year={2025},
      journal={Advances in Neural Information Processing Systems},
      eprint={2507.07136},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2507.07136}, 
}</code></pre>
  </div>
</section>

<section class="section" id="Research Series">
  <div class="container is-max-desktop content">
    <h2 class="title">Research Series</h2>
    <a href="https://langsplat.github.io/" target="_blank" style="text-decoration: underline;">[CVPR 2024] Langsplat: 3d language gaussian splatting</a>
    <pre><code>@inproceedings{qin2024langsplat,
  title={Langsplat: 3d language gaussian splatting},
  author={Qin, Minghan and Li, Wanhua and Zhou, Jiawei and Wang, Haoqian and Pfister, Hanspeter},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20051--20060},
  year={2024}
}</code></pre>
    <a href="https://4d-langsplat.github.io/" target="_blank" style="text-decoration: underline;">[CVPR 2025] 4d langsplat: 4d language gaussian splatting via multimodal large language models</a>
    <pre><code>@inproceedings{li20254d,
  title={4d langsplat: 4d language gaussian splatting via multimodal large language models},
  author={Li, Wanhua and Zhou, Renping and Zhou, Jiawei and Song, Yingwei and Herter, Johannes and Qin, Minghan and Huang, Gao and Pfister, Hanspeter},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={22001--22011},
  year={2025}
}</code></pre>
  </div>
</section>
