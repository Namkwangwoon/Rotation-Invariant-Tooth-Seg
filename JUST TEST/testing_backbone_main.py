from model.backbone_only import RIPointTransformer

from dataset.tdmatch import DentalMeshDataset
from torch.utils.data import DataLoader
import torch
from torch import optim
import torch.nn as nn
import os
import torch.optim.lr_scheduler as lr_scheduler
from scheduler import CosineLRScheduler
import numpy as np
import math

import loss_segmentation

from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    # writer = SummaryWriter()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Device:', device)  # 출력결과: cuda 
    print('Count of using GPUs:', torch.cuda.device_count())   #출력결과: 2 (0, 1 두개 사용하므로)
    print('Current cuda device:', torch.cuda.current_device())  # 출력결과: 0 (0, 1 중 앞의 GPU #0 의미)
    

    # model = RIPointTransformer(transformer_architecture=['self', 'cross', 'self', 'cross', 'self', 'cross'], with_cross_pos_embed=True, factor=1)
    # model.cuda()
    # _model = RIPointTransformer(transformer_architecture=['self', 'cross', 'self', 'cross', 'self', 'cross'], with_cross_pos_embed=True, factor=1)
    # _model.cuda()
    # model = nn.DataParallel(_model).to(device)

    # print(model)

    batch_size = 1
        
    training_set = DataLoader(DentalMeshDataset(split_with_txt_path='base_name_train_fold.txt', augmentation=True), batch_size=batch_size, shuffle=True, num_workers=0)
    validation_set = DataLoader(DentalMeshDataset(split_with_txt_path='base_name_test_fold.txt', augmentation=False), batch_size=batch_size, shuffle=False, num_workers=0)
    # training_set = DentalMeshDataset(split_with_txt_path='base_name_train_fold.txt')
    # validation_set = DentalMeshDataset(split_with_txt_path='base_name_test_fold.txt')

    epochs = 100

    # optimizer = optim.SGD(
    #     model.parameters(),
    #     lr = 0.01,
    #     momentum=0.9,
    #     weight_decay=0.0001
    # )
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs * 0.4), int(epochs * 0.6), int(epochs * 0.8)], gamma=0.1)
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

    # min_val_loss = math.inf
    # max_val_cls_acc = 0.0
    # max_val_mask_acc = 0.0

    for epoch in range(epochs):
        
        # train_loss = 0.0
        # model.train()

        print()
        print("Start Training!")
        for i, data in enumerate(training_set):
            
            # src_pcd, src_normals, src_feats, src_raw_pcd, labels = data
            # src_pcd, src_normals, src_feats, src_raw_pcd, labels = src_pcd[0], src_normals[0], src_feats[0], src_raw_pcd[0], labels[0]
            # src_o = torch.tensor([src_raw_pcd.shape[0]]).to(src_raw_pcd).int()
            # mask_labels = labels.clone()
            # mask_labels[mask_labels>=0] = 0
            
            _, _, _, _, labels = data
            mask_labels = labels.clone()
            mask_labels[mask_labels>=0] = 0
            
            print("-1 :", (labels==-1).sum().item(), "(", (labels==-1).sum().item() / labels.shape[1], ")"
                  "/ 0 :", (labels==0).sum().item(),
                  "/ 1 :", (labels==1).sum().item(),
                  "/ 2 :", (labels==2).sum().item(),
                  "/ 3 :", (labels==3).sum().item(),
                  "/ 4 :", (labels==4).sum().item(),
                  "/ 5 :", (labels==5).sum().item(),
                  "/ 6 :", (labels==6).sum().item(),
                  "/ 7 :", (labels==7).sum().item(),
                  "/ 8 :", (labels==8).sum().item(),
                  "/ 9 :", (labels==9).sum().item(),
                  "/ 10 :", (labels==10).sum().item(),
                  "/ 11 :", (labels==11).sum().item(),
                  "/ 12 :", (labels==12).sum().item(),
                  "/ 13 :", (labels==13).sum().item(),
                  "/ 14 :", (labels==14).sum().item(),
                  "/ 15 :", (labels==15).sum().item())
            
            
            
            # cls_output, mask_output = model([src_raw_pcd, src_feats, src_o, src_normals])
            
            # print()
            # print("src pcd :", src_pcd.shape)
            # print("src_normals :",src_normals.shape)
            # print("src_feats :", src_feats.shape)
            # print("src_raw_pcd :", src_raw_pcd.shape)
            # print("labels :", labels.shape)
            # print(labels.max())
            # print()
            
            # print("output :", output.shape)
            # print()
            
            # train_cls_loss = loss_segmentation.tooth_class_loss(cls_output, labels, 17)
            # train_mask_loss = loss_segmentation.tooth_class_loss(mask_output, mask_labels, 2)
            
            # train_total_loss = train_cls_loss + train_mask_loss
            # train_total_loss = train_mask_loss
            
            # optimizer.zero_grad()
            # train_total_loss.backward()
            # optimizer.step()
            
            # train_loss += train_total_loss
            
            
            # if (i+1) % 20 == 0:
            #     print("Epoch: [{}/{}][{}/{}]  Cls_loss: {train_cls_loss:.6f}  Mask_loss: {train_mask_loss:.6f}  Total_loss: {train_total_loss:.6f}"
            #         .format(epoch+1, epochs, i+1, len(training_set),
            #                 train_cls_loss=train_cls_loss.item(),
            #                 train_mask_loss=train_mask_loss.item(),
            #                 train_total_loss=train_total_loss.item()))
            # if (i+1) % 20 == 0:
            #     print("Epoch: [{}/{}][{}/{}]  Mask_loss: {train_mask_loss:.6f}  Total_loss: {train_total_loss:.6f}"
            #         .format(epoch+1, epochs, i+1, len(training_set),
            #                 train_mask_loss=train_mask_loss.item(),
            #                 train_total_loss=train_total_loss.item()))
            
        # scheduler.step()

            
        
        with torch.no_grad():
            
            # val_loss = 0.0
            # # val_acc = 0.0
            # val_total_cls_acc = 0.0
            # val_total_mask_acc = 0.0
            # model.eval()
                
            print()
            print("Start Validation!")
            for i, data in enumerate(validation_set):
                
                _, _, _, _, labels = data
                mask_labels = labels.clone()
                mask_labels[mask_labels>=0] = 0
                
                print("-1 :", (labels==-1).sum().item(), "(", (labels==-1).sum().item() / labels.shape[1], ")"
                    "/ 0 :", (labels==0).sum().item(),
                    "/ 1 :", (labels==1).sum().item(),
                    "/ 2 :", (labels==2).sum().item(),
                    "/ 3 :", (labels==3).sum().item(),
                    "/ 4 :", (labels==4).sum().item(),
                    "/ 5 :", (labels==5).sum().item(),
                    "/ 6 :", (labels==6).sum().item(),
                    "/ 7 :", (labels==7).sum().item(),
                    "/ 8 :", (labels==8).sum().item(),
                    "/ 9 :", (labels==9).sum().item(),
                    "/ 10 :", (labels==10).sum().item(),
                    "/ 11 :", (labels==11).sum().item(),
                    "/ 12 :", (labels==12).sum().item(),
                    "/ 13 :", (labels==13).sum().item(),
                    "/ 14 :", (labels==14).sum().item(),
                    "/ 15 :", (labels==15).sum().item())
                
                # src_pcd, src_normals, src_feats, src_raw_pcd, labels = data
                # src_pcd, src_normals, src_feats, src_raw_pcd, labels = src_pcd[0], src_normals[0], src_feats[0], src_raw_pcd[0], labels[0]
                # src_o = torch.tensor([src_raw_pcd.shape[0]]).to(src_raw_pcd).int()
                # mask_labels = labels.clone()
                # mask_labels[mask_labels>=0] = 0
                
                # cls_output, mask_output = model([src_raw_pcd, src_feats, src_o, src_normals])
                
                # val_cls_loss = loss_segmentation.tooth_class_loss(cls_output, labels, 17)
                # val_mask_loss = loss_segmentation.tooth_class_loss(mask_output, mask_labels, 2)
                
                # val_total_loss = val_cls_loss + val_mask_loss
                # # val_total_loss = val_mask_loss
                
                # val_loss += val_total_loss
                
                
                # val_cls_acc = (cls_output.argmax(-1) == (labels+1).reshape(-1)).sum() / len(labels)
                # val_mask_acc = (mask_output.argmax(-1) == (mask_labels+1).reshape(-1)).sum() / len(mask_labels)
                
                # val_total_cls_acc += val_cls_acc
                # val_total_mask_acc += val_mask_acc
                
                
                # if (i+1) % 20 == 0:
                #     print("Epoch: [{}/{}][{}/{}]  Cls_loss: {val_cls_loss:.6f}  Mask_loss: {val_mask_loss:.6f}  Total_loss: {val_total_loss:.6f}  |  Cls_acc: {val_cls_acc:.6f}  Mask_acc: {val_mask_acc:.6f}"
                #         .format(epoch+1, epochs, i+1, len(validation_set),
                #                 val_cls_loss=val_cls_loss.item(),
                #                 val_mask_loss=val_mask_loss.item(),
                #                 val_total_loss=val_total_loss.item(),
                #                 val_cls_acc=val_cls_acc.item(),
                #                 val_mask_acc=val_mask_acc.item()))
                # if (i+1) % 20 == 0:
                #     print("Epoch: [{}/{}][{}/{}]  Mask_loss: {val_mask_loss:.6f}  Total_loss: {val_total_loss:.6f}  |  Mask_acc: {val_mask_acc:.6f}"
                #         .format(epoch+1, epochs, i+1, len(validation_set),
                #                 val_mask_loss=val_mask_loss.item(),
                #                 val_total_loss=val_total_loss.item(),
                #                 val_mask_acc=val_mask_acc.item()))
        break
        
        # val_losses = val_loss / len(validation_set)
        # val_cls_accs = val_total_cls_acc / len(validation_set)
        # val_mask_accs = val_total_mask_acc / len(validation_set)
            
        # writer.add_scalar("Loss/Train", train_loss / len(training_set), epoch+1)
        # writer.add_scalar("Loss/Validation", val_losses, epoch+1)
        # writer.add_scalar("Loss/Class Accuracy", val_cls_accs, epoch+1)
        # writer.add_scalar("Loss/Mask Accuracy", val_mask_accs, epoch+1)
        
        # print("Epoch: [{}/{}]  Training Loss: {train_loss:.6f},  Validation Loss : {val_loss:.6f},  Validation Class Accuracy : {val_cls_accs:.6f},  Validation Mask Accuracy : {val_mask_accs:.6f}".format(epoch+1, epochs,
        #                                                                                                                                                                                                    train_loss=train_loss / len(training_set),
        #                                                                                                                                                                                                    val_loss=val_losses,
        #                                                                                                                                                                                                    val_cls_accs=val_cls_accs,
        #                                                                                                                                                                                                    val_mask_accs=val_mask_accs))
        # print("Epoch: [{}/{}]  Training Loss: {train_loss:.6f},  Validation Loss : {val_loss:.6f},  Validation Mask Accuracy : {val_mask_accs:.6f}".format(epoch+1, epochs,
        #                                                                                                                                                         train_loss=train_loss / len(training_set),
        #                                                                                                                                                         val_loss=val_losses,
        #                                                                                                                                                         val_mask_accs=val_mask_accs))
        
        
        # if min_val_loss > val_losses:
        #     print(f'Validation Loss Decreased({min_val_loss:.6f}--->{val_losses:.6f}) \t Saving The Model')
        #     min_val_loss = val_losses
        
        #     torch.save(model.state_dict(), f'checkpoints/epoch{epoch+1}_val{min_val_loss:.6f}.pth')
        
        # if max_val_acc < val_accs:
        #     print(f'Validation Accuracy Increaded({max_val_acc:.6f}--->{val_accs:.6f}) \t Saving The Model')
        #     max_val_acc = val_accs
            
        #     torch.save(model.state_dict(), f'checkpoints/epoch{epoch+1}_val{max_val_acc:.6f}.pth')
        
        # if min_val_loss > val_losses or max_val_mask_acc < val_mask_accs:
        #     print(f'Loss({min_val_loss:.6f}--->{val_losses:.6f})  Mask Accuracy({max_val_mask_acc:.6f}--->{val_mask_accs:.6f})\t Saving The Model')
            
        #     if min_val_loss > val_losses:
        #         min_val_loss = val_losses
        #     if max_val_mask_acc < val_mask_accs:
        #         max_val_mask_acc = val_mask_accs
            
        #     torch.save(model.state_dict(), f'checkpoints/epoch{epoch+1}_val{val_losses:.4f}_mask_acc{val_mask_accs:.4f}.pth')
        
        # if min_val_loss > val_losses or max_val_cls_acc < val_cls_accs or max_val_mask_acc < val_mask_accs:
        #     print(f'Loss({min_val_loss:.6f}--->{val_losses:.6f})  Class Accuracy({max_val_cls_acc:.6f}--->{val_cls_accs:.6f})  Mask Accuracy({max_val_mask_acc:.6f}--->{val_mask_accs:.6f})\t Saving The Model')
            
        #     if min_val_loss > val_losses:
        #         min_val_loss = val_losses
        #     if max_val_cls_acc < val_cls_accs:
        #         max_val_cls_acc = val_cls_accs
        #     if max_val_mask_acc < val_mask_accs:
        #         max_val_mask_acc = val_mask_accs
            
        #     torch.save(model.state_dict(), f'checkpoints/lims_without_norm_5_encdec_multiscale_cls_mask+cls_40/epoch{epoch+1}_val{val_losses:.4f}_cls_acc{val_cls_accs:.4f}_mask_acc{val_mask_accs:.4f}.pth')
            
            
            
            
    # writer.flush()
    # writer.close()