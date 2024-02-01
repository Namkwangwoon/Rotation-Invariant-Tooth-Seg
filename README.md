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
- Example

  <img src="https://github.com/Namkwangwoon/Rotation-Invariant-Tooth-Seg/assets/19163372/b5c181cf-0838-49a0-ad14-4bdc7ed8cc39" width="40%"/>


# Evaluation
```shell
python eval.py
```
### TSA
- Teeth Segmentation Accuracy, `'F1'`
- $precision={TP \over TP+FP},~~~~recall={TP \over TP+FN}$

  $$TSA = 2\times{precision\times recall \over precision+recall}$$
  
### TIR
- Teeth Identification Rate, `'ACC'`
  $$TIR={TP+TN \over TP+TN+FP+FN}$$
  
### IOU
- Intersection over Union, `'IOU'`
  $$IoU = {TP \over TP+FP+FN}$$

# Results
## Aligned Training
<img width="638" alt="image" src="https://github.com/Namkwangwoon/Rotation-Invariant-Tooth-Seg/assets/19163372/0b72d6c2-9907-4c46-94ba-70022d340556">

<img width="100%" alt="image" src="https://github.com/Namkwangwoon/Rotation-Invariant-Tooth-Seg/assets/19163372/511ca95a-f3ea-4313-a0b2-aeb32a836934">

## Random Rotated Training
<img width="634" alt="image" src="https://github.com/Namkwangwoon/Rotation-Invariant-Tooth-Seg/assets/19163372/4b0890b0-72a3-4c73-9391-269fea0c3381">

<img width="100%" alt="image" src="https://github.com/Namkwangwoon/Rotation-Invariant-Tooth-Seg/assets/19163372/6fc9df28-c6a2-46af-9a81-3c801016cd65">



# Other Methods
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
- https://github.com/yanx27/Pointnet_Pointnet2_pytorch
- https://github.com/POSTECH-CVLab/point-transformer
