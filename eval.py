import torch
from model.backbone_only import RIPointTransformer
from model.pointnet_sem_seg import get_model
from model.pointtransformer_seg import PointTransformerSeg, PointTransformerBlock

import os
import gen_utils as gu
import numpy as np

import pyvista as pv
import open3d as o3d

import pymeshlab
import augmentator as aug
from dataset.common import normal_redirect

import argparse
from dataset.tdmatch import DentalMeshDataset
from torch.utils.data import DataLoader

from evaluation import cal_metric

palet = np.array([
    [255,153,153],

    [153,76,0],
    [153,153,0],
    [76,153,0],
    [0,153,153],
    [0,0,153],
    [153,0,153],
    [153,0,76],
    [64,64,64],

    [20, 10, 0],
    [10, 10, 0],
    [10, 20, 0],
    [0, 10, 20],
    [0, 0, 20],
    [10, 0, 10],
    [10, 0, 0],
    [10, 10, 10],
    ])/255


Y_AXIS_MAX = 33.15232091532151
Y_AXIS_MIN = -36.9843781139949

view_point = np.array([0., 0., 0.])

aug_obj = aug.Augmentator([aug.Rotation([-180,180], 'rand')])

parser = argparse.ArgumentParser()
parser.add_argument('--model', default=None, type=str)
parser.add_argument('--lr_head', action='store_true')
args = parser.parse_args()



# checkpoint_path = 'checkpoints_rollback/simplification_saved_maskhead_multiscale/epoch77_val1.0147_cls_acc0.7454_mask_acc0.9737.pth'
# checkpoint_path = 'checkpoints_rollback/simplification_saved_maskhead_multiscale_vertexnorm/epoch83_val0.9012_cls_acc0.8159_mask_acc0.9616.pth'
# checkpoint_path = 'checkpoints_rollback/simplification_saved_maskhead_multiscale_vertexnorm_focal/epoch88_val0.9179_cls_acc0.7738_mask_acc0.9678_.pth'
# checkpoint_path = 'checkpoints_rollback/fps_saved/epoch62_val0.4737_cls_acc0.7195.pth'
checkpoint_path = 'checkpoints_rollback/simplification_saved/epoch65_val0.4956_cls_acc0.7171.pth'
# checkpoint_path = 'checkpoints_rollback/simplification_saved_maskhead/epoch79_val0.5562_cls_acc0.7180_mask_acc0.9841.pth'
# checkpoint_path = 'checkpoints_rollback/simplification_saved_maskhead_multiscale_focal/epoch85_val0.8152_cls_acc0.7560_mask_acc0.9757.pth'
# checkpoint_path = 'checkpoints_rollback/simplification_saved_maskhead_multiscale_vertexnorm_normredirect/epoch82_val0.7044_cls_acc0.8453_mask_acc0.9674_.pth'

# checkpoint_path = 'checkpoints_rotate/vertexnorm/epoch66_val0.8429_cls_acc0.7969_mask_acc0.9573_.pth'
# checkpoint_path = 'checkpoints_rotate/aligned_val/vertexnorm_normredir/epoch87_val0.7268_cls_acc0.8255_mask_acc0.9676_.pth'

## PointNet
# checkpoint_path = 'checkpoints_pointnet/vertexnorm/epoch97_val1.1880_cls_acc0.7050_mask_acc0.0000.pth'
# checkpoint_path = 'checkpoints_pointnet/rotate_aligned_vertexnorm/epoch58_val2.2611_cls_acc0.5248_mask_acc0.0000.pth'

## Point Transformer
# checkpoint_path = 'checkpoints_pointtransformer/vertexnorm/epoch90_val1.2929_cls_acc0.8183_mask_acc0.0000.pth'
# checkpoint_path = 'checkpoints_pointtransformer/rotate_aligned_vertexnorm/epoch74_val2.9434_cls_acc0.5932_mask_acc0.0000.pth'

print(checkpoint_path)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)  # 출력결과: cuda 
print('Count of using GPUs:', torch.cuda.device_count())   #출력결과: 2 (0, 1 두개 사용하므로)
print('Current cuda device:', torch.cuda.current_device())  # 출력결과: 0 (0, 1 중 앞의 GPU #0 의미)


if args.model == None:
    model = RIPointTransformer(transformer_architecture=['self', 'cross', 'self', 'cross', 'self', 'cross'], with_cross_pos_embed=True, factor=1)
elif args.model == 'pointnet':
    model = get_model(17)
elif args.model == 'pointtransformer':
    model = PointTransformerSeg(PointTransformerBlock, [2, 3, 4, 6, 3], c=1, k=17)
model.cuda()
model.load_state_dict(torch.load(checkpoint_path))
# model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
model.eval()

batch_size = 1

validation_set = DataLoader(DentalMeshDataset(split_with_txt_path='base_name_test_fold.txt', augmentation=False), batch_size=batch_size, shuffle=False, num_workers=0)

with torch.no_grad():
    
    val_loss = 0.0
    # val_acc = 0.0
    val_total_cls_acc = 0.0
    val_total_mask_acc = 0.0
    # val_total_sem_acc = 0.0
    val_total_lr_acc = 0.0
    
    val_total_iou = 0.0
    val_total_f1 = 0.0
    val_total_acc = 0.0
    val_total_sem_acc = 0.0
    
    model.eval()
        
    print()
    print("Start Validation!")
    for i, data in enumerate(validation_set):
        
        src_pcd, src_normals, src_feats, src_raw_pcd, labels = data
        src_pcd, src_normals, src_feats, src_raw_pcd, labels = src_pcd[0], src_normals[0], src_feats[0], src_raw_pcd[0], labels[0]
        src_o = torch.tensor([src_raw_pcd.shape[0]]).to(src_raw_pcd).int()
        labels = labels.reshape(-1)
        
        mask_labels = labels.clone()
        mask_labels[mask_labels>0] = 1
    
        sem_labels = labels.clone()
        sem_labels[sem_labels>8] = sem_labels[sem_labels>8] - 8
        
        lr_labels = labels.clone()
        lr_labels[(lr_labels>=1) & (lr_labels<=8)] = 1
        lr_labels[(lr_labels>=9) & (lr_labels<=16)] = 2
        
        
        if args.model == None:
            # cls_output, mask_output, sem_output = model([src_raw_pcd, src_feats, src_o, src_normals])
            cls_output, mask_output, lr_output = model([src_raw_pcd, src_feats, src_o, src_normals])
            # mask_output = mask_output.argmax(-1)
        elif args.model == 'pointnet':
            src_raw_pcd = src_raw_pcd.unsqueeze(0).permute(0, 2, 1)
            cls_output = model(src_raw_pcd)[0]
        elif args.model == 'pointtransformer':
            cls_output = model([src_raw_pcd, src_feats, src_o])
            
        cls_output = cls_output.argmax(-1)
        
        sem_output = cls_output.clone()
        sem_output[sem_output>8] = sem_output[sem_output>8] - 8
        
        iou, f1, acc, sem_acc, iou_arr = cal_metric(labels.detach().cpu(), cls_output.detach().cpu(), cls_output.detach().cpu())
        
        
        if (i+1) % 10 == 0:
            print("[{}/{}]  IOU: {iou:.6f}   F1: {f1:.6f}   ACC: {acc:.6f}   SEM_ACC: {sem_acc:.6f}"
                .format(i+1, len(validation_set), iou=iou, f1=f1, acc=acc, sem_acc=sem_acc))
            
        val_total_iou += iou
        val_total_f1 += f1
        val_total_acc += acc
        val_total_sem_acc += sem_acc

print()
print()
print("[Results]  IOU: {iou:.6f}   F1: {f1:.6f}   ACC: {acc:.6f}   SEM_ACC: {sem_acc:.6f}".format(iou=val_total_iou/len(validation_set),
                                                                                                  f1=val_total_f1/len(validation_set),
                                                                                                  acc=val_total_acc/len(validation_set),
                                                                                                  sem_acc=val_total_sem_acc/len(validation_set)))