# CoLaSplat

CoLaSplat implements our core algorithm for compressed scene representation and rendering. The main logic is in [`admm.py`](admm.py).

---

## Installation

Clone this repository. The required third-party dependencies are included under
`submodules/`:

```bash
git clone https://github.com/open-jian/3dgs-co-compress.git
cd 3dgs-co-compress
```

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate colasplat
```


## Quick Start
Use the demo compressed model to render:

```bash
python render_admm_quant.py \
  -s ../../Data/3dovs/bed \
  -m ../../Output/colasplat/admm_quant/3dovs/bed \
  --dataset 3dovs --include_feature
```





---

## Dataset

Please refer to the following repositories for the datasets:

* [3D-OVS](https://github.com/Kunhao-Liu/3D-OVS)
* [LERF](https://github.com/minghanqin/LangSplat)

The data should be organized as follows (example for the `bed` scene from 3D-OVS):

```
../../Data/
└── 3dovs/
    └── bed/
        ├── images/
        │   ├── img_0001.png
        │   ├── img_0002.png
        │   └── ...
        └── annotations/
            ├── ann_0001.json
            ├── ann_0002.json
            └── ...
```

We provide a demo of a compressed model trained on the `bed` scene. The point clouds and codebook files can be found at:

```
../../Output/colasplat/admm_quant/3dovs/bed/point_cloud/iteration_10000
```

For data preprocessing, please refer to the [LangSplat repository](https://github.com/minghanqin/LangSplat).

---


All generated models, renders, logs, and evaluation results are stored outside
the source tree under the workspace-level `Output/` directory. Set
`OUTPUT_ROOT` to override it. Compressed results default to:

```
../../Output/colasplat/admm_quant/
```

## Training process

### 1. Generate initial 3DGS point cloud

Semantic learning starts from a fully trained RGB 3DGS checkpoint, not only a
PLY. For the `bed` scene the retained 30k RGB model is:

```
../../Output/rgb_3dgs/3dovs/bed/chkpnt30000.pth
```

### 2. Semantic learning

Generate it using the provided script:

```bash
CoLaSplat/scripts/3dovs.sh
```

This corresponds to the first 30,000 iterations in the paper.

---

### 3. Compression

After semantic learning has produced
`../../Output/colasplat/semantic/3dovs/bed/chkpnt30000.pth`, run compression:

```bash
CoLaSplat/scripts/3dovs_admm_quant.sh
```

Results will be saved at:

```
../../Output/colasplat/admm_quant/3dovs/bed/train
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
