from model.pointtransformer_seg import PointTransformerSeg, PointTransformerBlock

from dataset.tdmatch import DentalMeshDataset, DentalMeshSampledDataset
from torch.utils.data import DataLoader
import torch
from torch import optim
import torch.nn as nn
import os
import torch.optim.lr_scheduler as lr_scheduler
from scheduler import CosineLRScheduler
import numpy as np
import math
from time import time

import loss_segmentation

from torch.utils.tensorboard import SummaryWriter
import argparse

# import open3d as o3d


# palet = torch.tensor([
#     [255,153,153],

#     [153,76,0],
#     [153,153,0],
#     [76,153,0],
#     [0,153,153],
#     [0,0,153],
#     [153,0,153],
#     [153,0,76],
#     [64,64,64],

#     [60, 30, 0],
#     [60, 60, 0],
#     [30, 60, 0],
#     [0, 60, 60],
#     [0, 0, 60],
#     [60, 0, 60],
#     [60, 0, 30],
#     [30, 30, 30],
#     ]).cuda()/255


parser = argparse.ArgumentParser()
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--lr_head', action='store_true')
args = parser.parse_args()

dir_name = 'rotate_rotate_vertexnorm'
print("<", dir_name, ">")

if not os.path.exists(os.path.join('checkpoints_pointtransformer', dir_name)):
    os.mkdir(os.path.join('checkpoints_pointtransformer', dir_name))

if __name__ == '__main__':
    writer = SummaryWriter(os.path.join('runs_pointtransformer', dir_name))

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Device:', device)  # 출력결과: cuda 
    print('Count of using GPUs:', torch.cuda.device_count())   #출력결과: 2 (0, 1 두개 사용하므로)
    print('Current cuda device:', torch.cuda.current_device())  # 출력결과: 0 (0, 1 중 앞의 GPU #0 의미)
    

    model = PointTransformerSeg(PointTransformerBlock, [2, 3, 4, 6, 3], c=1, k=17)
    model.cuda()

    batch_size = 1
        
    # training_set = DataLoader(DentalMeshDataset(split_with_txt_path='base_name_train_fold.txt', augmentation=True), batch_size=batch_size, shuffle=True, num_workers=0)
    # validation_set = DataLoader(DentalMeshDataset(split_with_txt_path='base_name_test_fold.txt', augmentation=False), batch_size=batch_size, shuffle=False, num_workers=0)
    training_set = DataLoader(DentalMeshSampledDataset(split_with_txt_path='base_name_train_fold.txt', augmentation=True), batch_size=batch_size, shuffle=True, num_workers=0)
    validation_set = DataLoader(DentalMeshSampledDataset(split_with_txt_path='base_name_test_fold.txt', augmentation=True), batch_size=batch_size, shuffle=False, num_workers=0)
    # training_set = DataLoader(DentalMeshSampledDataset(split_with_txt_path='base_name_train_fold_osstem.txt', augmentation=True), batch_size=batch_size, shuffle=True, num_workers=0)
    # validation_set = DataLoader(DentalMeshSampledDataset(split_with_txt_path='base_name_test_fold_osstem.txt', augmentation=False), batch_size=batch_size, shuffle=False, num_workers=0)

    start_epoch = 0
    epochs = 100

    optimizer = optim.SGD(
        model.parameters(),
        lr = 1e-2,
        momentum=0.9,
        weight_decay=0.0001
    )
    
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs * 0.6), int(epochs * 0.8)], gamma=0.1)
    
    
    if args.resume:
        checkpoint = torch.load(os.path.join('checkpoints_pointtransformer', dir_name, args.resume))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
        
    # shceduler = CosineLRScheduler(
    #         optimizer,
    #         t_initial=40,
    #         lr_min=1e-5,
    #         warmup_lr_init=1e-6,
    #         warmup_t=0,
    #         k_decay=1.0,
    #         cycle_mul=1,
    #         cycle_decay=0.1,
    #         cycle_limit=1,
    #         noise_range_t=None,
    #         noise_pct=0.67,
    #         noise_std=1.,
    #         noise_seed=42,
    #     )
    # criterion = nn.CrossEntropyLoss().cuda()
        

    min_val_loss = math.inf
    max_val_cls_acc = 0.0

    for epoch in range(start_epoch, epochs):
        
        train_loss = 0.0
        model.train()
    
        lr = optimizer.param_groups[0]['lr']

        print()
        print(f"Start Training!  lr : [{lr}]")
        for i, data in enumerate(training_set):
            start_time = time()
            
            src_pcd, src_normals, src_feats, src_raw_pcd, labels = data
            src_pcd, src_normals, src_feats, src_raw_pcd, labels = src_pcd[0], src_normals[0], src_feats[0], src_raw_pcd[0], labels[0]
            src_o = torch.tensor([src_raw_pcd.shape[0]]).to(src_raw_pcd).int()
            
            mask_labels = labels.clone()
            mask_labels[mask_labels>0] = 1
            
            
            lr_labels = labels.clone()
            lr_labels[(lr_labels>=1) & (lr_labels<=8)] = 1
            lr_labels[(lr_labels>=9) & (lr_labels<=16)] = 2
            
            
            cls_output = model([src_raw_pcd, src_feats, src_o])
            
            train_cls_loss = loss_segmentation.tooth_class_loss(cls_output, labels, 17)
            train_total_loss = train_cls_loss
            
            
            optimizer.zero_grad()
            train_total_loss.backward()
            optimizer.step()
            
            train_loss += train_total_loss
            
            
            if (i+1) % 20 == 0:
                print("Epoch: [{}/{}][{}/{}]  Cls_loss: {train_cls_loss:.6f}  Total_loss: {train_total_loss:.6f}"
                    .format(epoch+1, epochs, i+1, len(training_set),
                            train_cls_loss=train_cls_loss.item(),
                            train_total_loss=train_total_loss.item()))
                
            end_time = time()
            
        scheduler.step()
        
        with torch.no_grad():
            
            val_loss = 0.0
            val_total_cls_acc = 0.0
            val_total_mask_acc = 0.0
            val_total_lr_acc = 0.0
            model.eval()
                
            print()
            print("Start Validation!")
            for i, data in enumerate(validation_set):
                
                src_pcd, src_normals, src_feats, src_raw_pcd, labels = data
                src_pcd, src_normals, src_feats, src_raw_pcd, labels = src_pcd[0], src_normals[0], src_feats[0], src_raw_pcd[0], labels[0]
                src_o = torch.tensor([src_raw_pcd.shape[0]]).to(src_raw_pcd).int()
                
                mask_labels = labels.clone()
                mask_labels[mask_labels>0] = 1
            
                
                lr_labels = labels.clone()
                lr_labels[(lr_labels>=1) & (lr_labels<=8)] = 1
                lr_labels[(lr_labels>=9) & (lr_labels<=16)] = 2
                
                
                cls_output = model([src_raw_pcd, src_feats, src_o])
                
                val_cls_loss = loss_segmentation.tooth_class_loss(cls_output, labels, 17)
                
                
                val_total_loss = val_cls_loss
                
                val_loss += val_total_loss
                
                
                val_cls_acc = (cls_output.argmax(-1) == (labels).reshape(-1)).sum() / len(labels)
                
                val_total_cls_acc += val_cls_acc
                
                
                if (i+1) % 20 == 0:
                    print("Epoch: [{}/{}][{}/{}]  Cls_loss: {val_cls_loss:.6f}  Total_loss: {val_total_loss:.6f}  |  Cls_acc: {val_cls_acc:.6f}"
                        .format(epoch+1, epochs, i+1, len(validation_set),
                                val_cls_loss=val_cls_loss.item(),
                                val_total_loss=val_total_loss.item(),
                                val_cls_acc=val_cls_acc.item()))
        
        val_losses = val_loss / len(validation_set)
        val_cls_accs = val_total_cls_acc / len(validation_set)
        val_mask_accs = val_total_mask_acc / len(validation_set)
        if args.lr_head:
            val_lr_accs = val_total_lr_acc / len(validation_set)
            
        writer.add_scalar("Loss/Train", train_loss / len(training_set), epoch+1)
        writer.add_scalar("Loss/Validation", val_losses, epoch+1)
        writer.add_scalar("Loss/Class Accuracy", val_cls_accs, epoch+1)
        if args.lr_head:
            writer.add_scalar("Loss/Left-Right Accuracy", val_lr_accs, epoch+1)
        
        print("Epoch: [{}/{}]  Training Loss: {train_loss:.6f},  Validation Loss : {val_loss:.6f},  Validation Cls Accuracy : {val_cls_accs:.6f}".format(epoch+1, epochs,
                                                                                                                                                                train_loss=train_loss / len(training_set),
                                                                                                                                                                val_loss=val_losses,
                                                                                                                                                                val_cls_accs=val_cls_accs))
        
        if min_val_loss > val_losses or max_val_cls_acc < val_cls_accs:
            print(f'Loss({min_val_loss:.6f}--->{val_losses:.6f})  Class Accuracy({max_val_cls_acc:.6f}--->{val_cls_accs:.6f})\t Saving The Model')
            
            if min_val_loss > val_losses:
                min_val_loss = val_losses
            if max_val_cls_acc < val_cls_accs:
                max_val_cls_acc = val_cls_accs
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_losses,
                'cls_acc': val_cls_accs,
                'mask_accs': val_mask_accs},
                f'checkpoints_pointtransformer/{dir_name}/epoch{epoch+1}_val{val_losses:.4f}_cls_acc{val_cls_accs:.4f}_mask_acc{val_mask_accs:.4f}.pth')            
            
        
            
            
            
            
    writer.flush()
    writer.close()