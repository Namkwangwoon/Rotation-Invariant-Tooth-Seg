from model.backbone_only import AxesRegressor, RIPointTransformerAxesRegressor
from model.pointtransformer_seg import PointTransformerAxesRegressor

from torch.utils.tensorboard import SummaryWriter
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
import math
from dataset.tdmatch import DentalMeshDatasetAxes

dict_name = 'NxMSE_lr1e-5_adamcosine_PTencoder_manyfc_lessaug'

if not os.path.exists(os.path.join('checkpoints_axes_regression', dict_name)):
    os.makedirs(os.path.join('checkpoints_axes_regression', dict_name))

writer = SummaryWriter(os.path.join('runs_axes_regression',dict_name))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)  # 출력결과: cuda 
print('Count of using GPUs:', torch.cuda.device_count())   #출력결과: 2 (0, 1 두개 사용하므로)
print('Current cuda device:', torch.cuda.current_device())  # 출력결과: 0 (0, 1 중 앞의 GPU #0 의미)

# model = AxesRegressor(6)                      ## input dim: 6 (vertex coordinates(3) + vertex normals coordinates(3))
# model = RIPointTransformerAxesRegressor()     ## Rotation-Invariant encdoer + Linear layers
model = PointTransformerAxesRegressor()         ## Rotation-Variant encoder + Linear Layers
model.cuda()

batch_size = 1

training_set = DataLoader(DentalMeshDatasetAxes(mode='train', augmentation=True), batch_size=batch_size, shuffle=True, num_workers=0)
validation_set = DataLoader(DentalMeshDatasetAxes(mode='val', augmentation=True), batch_size=batch_size, shuffle=False, num_workers=0)

epochs = 100

# optimizer = optim.SGD(
#     model.parameters(),
#     lr = 1e-2,
#     momentum=0.9,
#     weight_decay=0.0001
# )
# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs * 0.6), int(epochs * 0.8)], gamma=0.1)
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-7)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

loss = nn.MSELoss()
# loss = nn.MSELoss(reduction='sum')

min_val_loss = math.inf

for epoch in range(epochs):
    
    train_epoch_loss = 0.0
    model.train()
    
    lr = optimizer.param_groups[0]['lr']

    print()
    print(f"Start Training!  lr : [{lr}]")
    for i, data in enumerate(training_set):
        
        src_pcd, src_normals, src_feats, src_raw_pcd, labels = data
        src_pcd, src_normals, src_feats, src_raw_pcd, labels = src_pcd[0], src_normals[0], src_feats[0], src_raw_pcd[0], labels[0]
        src_o = torch.tensor([src_raw_pcd.shape[0]]).to(src_raw_pcd).int()
        
        # output = model(torch.cat([src_pcd, src_normals], dim=-1))         ## Linear layers
        # output = model([src_raw_pcd, src_feats, src_o, src_normals])      ## Rotation-Invariant Encoder
        output = model([src_raw_pcd, src_feats, src_o])                     ## Rotation-Variant Encoder
        
        labels = labels.repeat(output.shape[0], 1)
        train_loss = loss(output, labels)
        train_epoch_loss += train_loss
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print("Epoch: [{}/{}][{}/{}]  MSE_loss: {train_loss:.6f}".format(epoch+1, epochs, i+1, len(training_set), train_loss=train_loss))
        
        
    scheduler.step()
    
    with torch.no_grad():
        
        val_epoch_loss = 0.0
        model.eval()
            
        print()
        print("Start Validation!")
        for i, data in enumerate(validation_set):
            
            src_pcd, src_normals, src_feats, src_raw_pcd, labels = data
            src_pcd, src_normals, src_feats, src_raw_pcd, labels = src_pcd[0], src_normals[0], src_feats[0], src_raw_pcd[0], labels[0]
            src_o = torch.tensor([src_raw_pcd.shape[0]]).to(src_raw_pcd).int()
            
            # output = model(torch.cat([src_pcd, src_normals], dim=-1))         ## Linear layers
            # output = model([src_raw_pcd, src_feats, src_o, src_normals])      ## Rotation-Invariant Encoder
            output = model([src_raw_pcd, src_feats, src_o])                     ## Rotation-Variant Encoder
            
            labels = labels.repeat(output.shape[0], 1)
            val_loss = loss(output, labels)
            val_epoch_loss += val_loss
            
            if (i+1) % 10 == 0:
                print("Epoch: [{}/{}][{}/{}]  MSE_loss: {val_loss:.6f}".format(epoch+1, epochs, i+1, len(validation_set), val_loss=val_loss))


    train_epoch_loss /= len(training_set)
    val_epoch_loss /= len(validation_set)
        
    writer.add_scalar("Loss/Train", train_epoch_loss, epoch+1)
    writer.add_scalar("Loss/Validation", val_epoch_loss, epoch+1)
    
    print("Epoch: [{}/{}]  Training Loss: {train_loss:.6f},  Validation Loss : {val_loss:.6f},".format(epoch+1, epochs,
                                                                                                       train_loss=train_epoch_loss,
                                                                                                       val_loss=val_epoch_loss))
        
        
    if min_val_loss > val_epoch_loss:
        print(f'Val Loss({min_val_loss:.6f} ---> {val_epoch_loss:.6f})\t Saving The Model')
        
        if min_val_loss > val_epoch_loss:
            min_val_loss = val_epoch_loss
        
        torch.save(model.state_dict(), f'checkpoints_axes_regression/{dict_name}/epoch{epoch+1}_val{val_epoch_loss:.4f}.pth')
        
        
writer.flush()
writer.close()