# Rotation Invariant Tooth Scan Segmentation
# Overall Architecture
<img width="100%" alt="image" src="https://github.com/Namkwangwoon/Rotation-Invariant-Tooth-Seg/assets/19163372/153099f6-ea6f-4146-9e7a-df2cfe7d0ab0">

# Training
```shell
python main.py \
  [--resume "checkpoint path"] \
  [--lr_head]
```
# Inference (Visualization)
## Our Method
```shell
python inference.py
```
## PointNet
```shell
python inference.py --model pointnet
```
## Point Transformer
```shell
python inference.py --model pointtransformer
```
# Evaluation
## Our Method
```shell
python eval.py
```
## PointNet
```shell
python eval.py --model pointnet
```
## PointTransformer
```shell
python eval.py --model pointttransformer
```
# Results
## Our Method
## PointNet
## PointTransformer
