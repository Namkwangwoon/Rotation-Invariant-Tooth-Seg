# Rotation Invariant Tooth Scan Segmentation
"3D Teeth Scan Segmentation via Rotation-Invariant Descriptor"

# Overall Architecture
![Overall](https://github.com/Namkwangwoon/Rotation-Invariant-Tooth-Seg/assets/19163372/31db7fd2-1efc-43e4-81cd-e33773d5338f)


# Prerequisites
- torch 1.8.0
- requirements
  ```shell
  pip install -r requirements.txt
  ```

# Training

- Before training, You can load pre-sampled dataset(.npy), or sample every epoch
  - For pre-sampling, see `"./preprocessing.ipynb"`

```shell
python main.py \
  [--resume "checkpoint path"]
```
 
- You can choose one of three sampling methods.
    1. Farthest Point Sampling(FPS)
    2. Poisson Disk Sampling (Creating new vertices)
    3. **Poisson Disk based Simplification (Keeping existing vertices, Using this sampling method)**

# Inference (Visualization)
```shell
python inference.py
```

- Visualization using `PyVista` remehsing, and `Open3D` visualization
- Color gingiva and each number of teeth a different color.
- example

  <img src="https://github.com/Namkwangwoon/Rotation-Invariant-Tooth-Seg/assets/19163372/b5c181cf-0838-49a0-ad14-4bdc7ed8cc39" width="40%"/>


# Evaluation
```shell
python eval.py
```
- IOU
- F1
- ACC
- SEM_ACC

# Results

# Comparison
## PointNet
### Training
```shell
python main_pointnet.py \
  [--resume "checkpoint path"]
```
### Inference
```shell
python inference.py --model pointnet
```
### Evaluation
```shell
python eval.py --model pointnet
```

## Point Transformer
### Training
```shell
python main_pointtransformer.py \
  [--resume "checkpoint path"]
```
### Inference
```shell
python inference.py --model pointtransformer
```
### Evaluation
```shell
python eval.py --model pointtransformer
```

# Reference Codes
- https://github.com/limhoyeon/ToothGroupNetwork.git
- https://github.com/haoyu94/RoITr.git
