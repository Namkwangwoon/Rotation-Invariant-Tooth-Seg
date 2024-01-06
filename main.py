from model.backbone_only import RIPointTransformer

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


parser = argparse.ArgumentParser()
parser.add_argument('--resume', default=None, type=str)     ## "python backbone_main.py --resume ~~~.pth" 시 dir_name의 ~~~.pth 부터 resume 학습
parser.add_argument('--lr_head', action='store_true')       ## "python lr_head" 시 lr_head가 포함된 구조의 학습 진행 (치아 좌우 구별을 잘 못해서 도움 줄려고 넣었었는데, 성능이 별 차이 없어서 결국 안씀)
args = parser.parse_args()

dir_name = 'vertexnorm_normredir_LRhead_focal'      ## checkpoint 및 tensorboard 저장할 디렉토리 이름
print("<", dir_name, ">")

if not os.path.exists(os.path.join('checkpoints_rotate', dir_name)):
    os.mkdir(os.path.join('checkpoints_rotate', dir_name))

if __name__ == '__main__':
    writer = SummaryWriter(os.path.join('runs_rotate', dir_name))


    ### GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Device:', device)  # 출력결과: cuda 
    print('Count of using GPUs:', torch.cuda.device_count())   #출력결과: 2 (0, 1 두개 사용하므로)
    print('Current cuda device:', torch.cuda.current_device())  # 출력결과: 0 (0, 1 중 앞의 GPU #0 의미)
    ###
    

    model = RIPointTransformer(transformer_architecture=['self', 'cross', 'self', 'cross', 'self', 'cross'], with_cross_pos_embed=True, factor=1)
    model.cuda()

    batch_size = 1
    
    ## 매 epoch마다 샘플링 하는 DataLoader
    # training_set = DataLoader(DentalMeshDataset(split_with_txt_path='base_name_train_fold.txt', augmentation=True), batch_size=batch_size, shuffle=True, num_workers=0)
    # validation_set = DataLoader(DentalMeshDataset(split_with_txt_path='base_name_test_fold.txt', augmentation=False), batch_size=batch_size, shuffle=False, num_workers=0)

    ## 미리 샘플링된 데이터가 저장된 .npy 파일들을 불러와서 학습 (샘플링하여 저장하는 코드는 "preprocessing.ipynb"의 "generate_simplified_point_cloud" 참고)
    training_set = DataLoader(DentalMeshSampledDataset(split_with_txt_path='base_name_train_fold.txt', augmentation=True), batch_size=batch_size, shuffle=True, num_workers=0)
    validation_set = DataLoader(DentalMeshSampledDataset(split_with_txt_path='base_name_test_fold.txt', augmentation=True), batch_size=batch_size, shuffle=False, num_workers=0)
    
    ## 미리 샘플링된 Osstem dataset 로딩
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
    
    
    ## "python backbone_main.py --resume ~~~.pth"시 checkpoint 로딩
    ## (중간에 추가한 기능이라, 일부 checkpoint에서 동작 안할수도 있음!)
    if args.resume:
        checkpoint = torch.load(os.path.join('checkpoints_rotate', dir_name, args.resume))
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
        

    ## (Minimum loss or Maximum accuracy) 갱신되면 checkpoint 저장
    min_val_loss = math.inf
    max_val_cls_acc = 0.0
    max_val_mask_acc = 0.0
    # max_val_sem_acc = 0.0
    max_val_lr_acc = 0.0
    ##
    

    for epoch in range(start_epoch, epochs):
        
        train_loss = 0.0        ## 한 epoch의 train loss
        model.train()
    
        lr = optimizer.param_groups[0]['lr']

        print()
        print(f"Start Training!  lr : [{lr}]")
        for i, data in enumerate(training_set):
            start_time = time()
            
            
            ## src_pcd      : 데이터의 3차원 vertices 좌표
            ## src_normals  : 데이터의 3차원 vertex normals 좌표
            ## src_feats    : 처음에는 1로 초기화 해놓고 model 안에서 (n, 3) -> (n, 64) -> (n, 128) 이런식으로 처리할 feature들
            ## src_raw_pcd  : src_pcd랑 비슷한데 실제 model input으로 줄 것들
            src_pcd, src_normals, src_feats, src_raw_pcd, labels = data
            src_pcd, src_normals, src_feats, src_raw_pcd, labels = src_pcd[0], src_normals[0], src_feats[0], src_raw_pcd[0], labels[0]
            ## src_o        : vertex 갯수
            src_o = torch.tensor([src_raw_pcd.shape[0]]).to(src_raw_pcd).int()
            
            
            ## mask branch(잇몸 0, 치아 1 을 분할하는 별도의 classification branch)를 위한 label
            mask_labels = labels.clone()
            mask_labels[mask_labels>0] = 1
            
            # lr_labels = labels.clone()
            # lr_labels[(lr_labels>=1) & (lr_labels<=8)] = 1
            # lr_labels[(lr_labels>=9) & (lr_labels<=16)] = 2
            
            
            # cls_output = model([src_raw_pcd, src_feats, src_o, src_normals])
            cls_output, mask_output, lr_output = model([src_raw_pcd, src_feats, src_o, src_normals])
            
            
            
            train_cls_loss = loss_segmentation.tooth_class_loss(cls_output, labels, 17)
            train_mask_loss = loss_segmentation.tooth_class_loss(mask_output, mask_labels, 2)
            # train_mask_loss = loss_segmentation.tooth_class_loss_focal(mask_output, mask_labels)      ## Focal loss
            if args.lr_head:
                train_lr_loss = loss_segmentation.tooth_class_loss(lr_output, lr_labels, 3)
            
            
            if args.lr_head:
                train_total_loss = train_cls_loss + train_mask_loss + train_lr_loss
            else:
                train_total_loss = train_cls_loss + train_mask_loss
            
            
            optimizer.zero_grad()
            train_total_loss.backward()
            optimizer.step()
            
            train_loss += train_total_loss
            
            
            # if (i+1) % 20 == 0:
            #     print("Epoch: [{}/{}][{}/{}]  Cls_loss: {train_cls_loss:.6f}  Total_loss: {train_total_loss:.6f}"
            #         .format(epoch+1, epochs, i+1, len(training_set),
            #                 train_cls_loss=train_cls_loss.item(),
            #                 train_total_loss=train_total_loss.item()))
            if args.lr_head:
                if (i+1) % 10 == 0:
                    print("Epoch: [{}/{}][{}/{}]  Cls_loss: {train_cls_loss:.6f}  Mask_loss: {train_mask_loss:.6f}  LR_loss: {train_lr_loss:.6f}  |  Total_loss: {train_total_loss:.6f}"
                        .format(epoch+1, epochs, i+1, len(training_set),
                                train_cls_loss=train_cls_loss.item(),
                                train_mask_loss=train_mask_loss.item(),
                                train_lr_loss=train_lr_loss.item(),
                                train_total_loss=train_total_loss.item()))
            else:
                if (i+1) % 10 == 0:
                    print("Epoch: [{}/{}][{}/{}]  Cls_loss: {train_cls_loss:.6f}  Mask_loss: {train_mask_loss:.6f}  |  Total_loss: {train_total_loss:.6f}"
                        .format(epoch+1, epochs, i+1, len(training_set),
                                train_cls_loss=train_cls_loss.item(),
                                train_mask_loss=train_mask_loss.item(),
                                train_total_loss=train_total_loss.item()))
                
            end_time = time()
            
        scheduler.step()
        
        
        with torch.no_grad():
            
            val_loss = 0.0
            # val_acc = 0.0
            val_total_cls_acc = 0.0
            val_total_mask_acc = 0.0
            # val_total_sem_acc = 0.0
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
                
                
                # cls_output = model([src_raw_pcd, src_feats, src_o, src_normals])
                cls_output, mask_output, lr_output = model([src_raw_pcd, src_feats, src_o, src_normals])
                
                val_cls_loss = loss_segmentation.tooth_class_loss(cls_output, labels, 17)
                val_mask_loss = loss_segmentation.tooth_class_loss(mask_output, mask_labels, 2)
                # val_mask_loss = loss_segmentation.tooth_class_loss_focal(mask_output, mask_labels)        ## Focal Loss
                if args.lr_head:
                    val_lr_loss = loss_segmentation.tooth_class_loss(lr_output, lr_labels, 3)
                
                # val_total_loss = val_cls_loss
                val_total_loss = val_cls_loss + val_mask_loss
                # val_total_loss = val_cls_loss + val_mask_loss + val_lr_loss
                
                val_loss += val_total_loss
                
                
                ### Accuracy 계산
                val_cls_acc = (cls_output.argmax(-1) == (labels).reshape(-1)).sum() / len(labels)
                val_mask_acc = (mask_output.argmax(-1) == (mask_labels).reshape(-1)).sum() / len(mask_labels)
                if args.lr_head:
                    val_lr_acc = (lr_output.argmax(-1) == (lr_labels).reshape(-1)).sum() / len(lr_labels)
                
                val_total_cls_acc += val_cls_acc
                val_total_mask_acc += val_mask_acc
                if args.lr_head:
                    val_total_lr_acc += val_lr_acc
                
                
                ### 10 epoch마다 결과 찍는 코드
                # if (i+1) % 20 == 0:
                #     print("Epoch: [{}/{}][{}/{}]  Cls_loss: {val_cls_loss:.6f}  Total_loss: {val_total_loss:.6f}  |  Cls_acc: {val_cls_acc:.6f}"
                #         .format(epoch+1, epochs, i+1, len(validation_set),
                #                 val_cls_loss=val_cls_loss.item(),
                #                 val_total_loss=val_total_loss.item(),
                #                 val_cls_acc=val_cls_acc.item()))
                if args.lr_head:
                    if (i+1) % 10 == 0:
                        print("Epoch: [{}/{}][{}/{}]  Cls_loss: {val_cls_loss:.6f}  Mask_loss: {val_mask_loss:.6f}  LR_loss: {val_lr_loss:.6f}    Total_loss: {val_total_loss:.6f}  |  Cls_acc: {val_cls_acc:.6f}  Mask_acc: {val_mask_acc:.6f}  LR_acc: {val_lr_acc:.6f}"
                            .format(epoch+1, epochs, i+1, len(validation_set),
                                    val_cls_loss=val_cls_loss.item(),
                                    val_mask_loss=val_mask_loss.item(),
                                    val_lr_loss=val_lr_loss.item(),
                                    val_total_loss=val_total_loss.item(),
                                    val_cls_acc=val_cls_acc.item(),
                                    val_mask_acc=val_mask_acc.item(),
                                    val_lr_acc=val_lr_acc.item()))
                else:
                    if (i+1) % 10 == 0:
                        print("Epoch: [{}/{}][{}/{}]  Cls_loss: {val_cls_loss:.6f}  Mask_loss: {val_mask_loss:.6f}    Total_loss: {val_total_loss:.6f}  |  Cls_acc: {val_cls_acc:.6f}  Mask_acc: {val_mask_acc:.6f}"
                            .format(epoch+1, epochs, i+1, len(validation_set),
                                    val_cls_loss=val_cls_loss.item(),
                                    val_mask_loss=val_mask_loss.item(),
                                    val_total_loss=val_total_loss.item(),
                                    val_cls_acc=val_cls_acc.item(),
                                    val_mask_acc=val_mask_acc.item()))
                ###
                
        
        val_losses = val_loss / len(validation_set)
        val_cls_accs = val_total_cls_acc / len(validation_set)
        val_mask_accs = val_total_mask_acc / len(validation_set)
        if args.lr_head:
            val_lr_accs = val_total_lr_acc / len(validation_set)
            
        writer.add_scalar("Loss/Train", train_loss / len(training_set), epoch+1)
        writer.add_scalar("Loss/Validation", val_losses, epoch+1)
        writer.add_scalar("Loss/Class Accuracy", val_cls_accs, epoch+1)
        writer.add_scalar("Loss/Mask Accuracy", val_mask_accs, epoch+1)
        if args.lr_head:
            writer.add_scalar("Loss/Left-Right Accuracy", val_lr_accs, epoch+1)
        
        # print("Epoch: [{}/{}]  Training Loss: {train_loss:.6f},  Validation Loss : {val_loss:.6f},  Validation Cls Accuracy : {val_cls_accs:.6f}".format(epoch+1, epochs,
        #                                                                                                                                                         train_loss=train_loss / len(training_set),
        #                                                                                                                                                         val_loss=val_losses,
        #                                                                                                                                                         val_cls_accs=val_cls_accs))
        
        
        ### Epoch별 결과 확인 
        if args.lr_head:
            print("Epoch: [{}/{}]  Training Loss: {train_loss:.6f},  Validation Loss : {val_loss:.6f},  Validation Class Accuracy : {val_cls_accs:.6f},  Validation Mask Accuracy : {val_mask_accs:.6f},  Validation Left-Right Accuracy : {val_lr_accs:.6f}".format(epoch+1, epochs,
                                                                                                                                                                                                               train_loss=train_loss / len(training_set),
                                                                                                                                                                                                               val_loss=val_losses,
                                                                                                                                                                                                               val_cls_accs=val_cls_accs,
                                                                                                                                                                                                               val_mask_accs=val_mask_accs,
                                                                                                                                                                                                               val_lr_accs=val_lr_accs))
 
        else:
            print("Epoch: [{}/{}]  Training Loss: {train_loss:.6f},  Validation Loss : {val_loss:.6f},  Validation Class Accuracy : {val_cls_accs:.6f},  Validation Mask Accuracy : {val_mask_accs:.6f}".format(epoch+1, epochs,
                                                                                                                                                                                                            train_loss=train_loss / len(training_set),
                                                                                                                                                                                                            val_loss=val_losses,
                                                                                                                                                                                                            val_cls_accs=val_cls_accs,
                                                                                                                                                                                                            val_mask_accs=val_mask_accs))
        ###
        
        
        ### (Minimum loss or Maximum accuracy) 갱신시 checkpoint 저장
        
        # if min_val_loss > val_losses or max_val_cls_acc < val_cls_accs:
        #     print(f'Loss({min_val_loss:.6f}--->{val_losses:.6f})  Class Accuracy({max_val_cls_acc:.6f}--->{val_cls_accs:.6f})\t Saving The Model')
            
        #     if min_val_loss > val_losses:
        #         min_val_loss = val_losses
        #     if max_val_cls_acc < val_cls_accs:
        #         max_val_cls_acc = val_cls_accs
            
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'loss': val_losses,
            #     'cls_acc': val_cls_accs,
            #     'mask_accs': val_mask_accs},
            #     f'checkpoints_osstem/{dir_name}/epoch{epoch+1}_val{val_losses:.4f}_cls_acc{val_cls_accs:.4f}_mask_acc{val_mask_accs:.4f}.pth')
            
            
        if args.lr_head: 
            if min_val_loss > val_losses or max_val_cls_acc < val_cls_accs or max_val_mask_acc < val_mask_accs or max_val_lr_acc < val_lr_accs:
                print(f'Loss({min_val_loss:.6f}--->{val_losses:.6f})  Class Accuracy({max_val_cls_acc:.6f}--->{val_cls_accs:.6f})  Mask Accuracy({max_val_mask_acc:.6f}--->{val_mask_accs:.6f})  Left-Right Accuracy({max_val_lr_acc:.6f}--->{val_lr_accs:.6f})\t Saving The Model')
                
                if min_val_loss > val_losses:
                    min_val_loss = val_losses
                if max_val_cls_acc < val_cls_accs:
                    max_val_cls_acc = val_cls_accs
                if max_val_mask_acc < val_mask_accs:
                    max_val_mask_acc = val_mask_accs
                if max_val_lr_acc < val_lr_accs:
                    max_val_lr_acc = val_lr_accs
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': val_losses,
                    'cls_acc': val_cls_accs,
                    'mask_accs': val_mask_accs},
                    f'checkpoints_rotate/{dir_name}/epoch{epoch+1}_val{val_losses:.4f}_cls_acc{val_cls_accs:.4f}_mask_acc{val_mask_accs:.4f}_LR_acc{val_lr_accs:.4f}.pth')
        
        else:
            if min_val_loss > val_losses or max_val_cls_acc < val_cls_accs or max_val_mask_acc < val_mask_accs:
                print(f'Loss({min_val_loss:.6f}--->{val_losses:.6f})  Class Accuracy({max_val_cls_acc:.6f}--->{val_cls_accs:.6f})  Mask Accuracy({max_val_mask_acc:.6f}--->{val_mask_accs:.6f})\t Saving The Model')
                
                if min_val_loss > val_losses:
                    min_val_loss = val_losses
                if max_val_cls_acc < val_cls_accs:
                    max_val_cls_acc = val_cls_accs
                if max_val_mask_acc < val_mask_accs:
                    max_val_mask_acc = val_mask_accs
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': val_losses,
                    'cls_acc': val_cls_accs,
                    'mask_accs': val_mask_accs},
                    f'checkpoints_rotate/{dir_name}/epoch{epoch+1}_val{val_losses:.4f}_cls_acc{val_cls_accs:.4f}_mask_acc{val_mask_accs:.4f}_.pth')
        ###
        
            
    writer.flush()
    writer.close()