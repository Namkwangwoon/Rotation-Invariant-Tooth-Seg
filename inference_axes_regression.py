from model.backbone_only import AxesRegressor

from torch.utils.tensorboard import SummaryWriter
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
import math
from dataset.tdmatch import DentalMeshDatasetAxes

from pytorch3d.transforms import so3_log_map, so3_exponential_map
import trimesh
import numpy as np
import augmentator as aug


checkpoint_path = 'checkpoints_axes_regression'

loss = nn.MSELoss()

model = AxesRegressor(6)        ## input dim: 6 (vertex coordinates(3) + vertex normals coordinates(3))
model.cuda()
model.load_state_dict(torch.load(checkpoint_path+'/NxMSE_lr1e-2/epoch96_val0.1524.pth'))
model.eval()
print(model)

source_path = '../datasets/osttemorigin'
cases = ['Case_01', 'Case_02', 'Case_03', 'Case_04', 'Case_05', 'Case_06', 'Case_07', 'Case_08', 'Case_09', 'Case_10']      ## Test(Validation) Cases
stl_path_ls = []
for case in cases:
    file_list = os.listdir(os.path.join(source_path, case, 'STL'))
    stl_path_ls += [os.path.join(source_path, case, 'STL', file) for file in file_list if file.endswith('.stl')]

group1 = ['Case_01', 'Case_03', 'Case_04']
group2 = ['Case_02', 'Case_05', 'Case_06', 'Case_07', 'Case_08', 'Case_09', 'Case_10']

gt_mat1 = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
gt_mat2 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])


### Random Imput Axes Augmentation
# aug_obj = aug.Augmentator([aug.Scaling([0.85, 1.15]), aug.Rotation([-30,30], 'fixed'), aug.Translation([-0.2, 0.2])])
aug_obj = aug.Augmentator([aug.Scaling([0.85, 1.15]), aug.Rotation([-180,180], 'rand'), aug.Translation([-0.2, 0.2])])


with torch.no_grad():
        
    val_epoch_loss = 0.0
    model.eval()

    for i, path in enumerate(stl_path_ls):
        case = stl_path_ls[i].split('/')[-3]
        if 'lower' in stl_path_ls[i]:
            jaw = 'lower'
        else:
            jaw = 'upper'
            
        ### Mesh Loading & get Vertices, Vertex Normals
        loaded_mesh = trimesh.load_mesh(stl_path_ls[i])
        vertices = loaded_mesh.vertices
        normals = loaded_mesh.vertex_normals
        ###
        
        vertices -= np.mean(vertices, axis=0)
        
        src_pcd = np.concatenate([vertices, normals], axis=-1)
        
        
        ### Random Rotation Augmentation
        aug_obj.reload_vals()
        src_pcd, aug_mats = aug_obj.run(src_pcd)
        ###
        
        loaded_mesh.vertices = src_pcd[:, :3]
        loaded_mesh.vertex_normals = src_pcd[:, 3:6]
        # loaded_mesh.export('input_'+stl_path_ls[i].split('/')[-1])      ### Random Rotate된 Mesh를 저장 (확인용)
        
        
        ### Random Rotation Augmentation GT Creating
        if ((case in group1) & (jaw=='lower')) | ((case in group2) & (jaw=='upper')):
            gt_mat = gt_mat1
        else:
            gt_mat = gt_mat2
            
        label_mat_pi = np.matmul(np.linalg.inv(aug_mats[1]).T, gt_mat)

        label = so3_log_map(torch.tensor([label_mat_pi]))[0] / np.pi
        ###
        
        
        src_pcd = torch.tensor(src_pcd).cuda().type(torch.float32)
        
        
        output = model(src_pcd)
        
        val_loss = loss(output, label.cuda().repeat(output.shape[0], 1))
        val_epoch_loss += val_loss
        
        ### Output(-1 ~ 1)  =>  Radian(-pi ~ pi)
        output_pi = (output * np.pi).mean(0)
        output_mat = so3_exponential_map(output_pi.unsqueeze(0))
        
        print()
        print("label_pi : ", label * np.pi)
        print("output_pi : ", output_pi)
        
        
        loaded_mesh.vertices = torch.matmul(src_pcd[:, :3].detach(), torch.tensor(output_mat[0]).detach().float()).cpu()
        loaded_mesh.vertex_normals = torch.matmul(src_pcd[:, 3:].detach(), torch.tensor(output_mat[0]).detach().float()).cpu()
        # loaded_mesh.vertices = torch.matmul(src_pcd[:, :3].detach(), so3_exponential_map((label*np.pi).unsqueeze(0).cuda())[0].float()).cpu()
        # loaded_mesh.vertex_normals = torch.matmul(src_pcd[:, 3:].detach(), so3_exponential_map((label*np.pi).unsqueeze(0).cuda())[0].float()).cpu()
        loaded_mesh.export('output_'+stl_path_ls[i].split('/')[-1])         ## Output 결과로 Mesh를 돌려서 저장
        
        
        print("Epoch: [{}/{}]  MSE_loss: {val_loss:.6f}".format(i+1, 20, val_loss=val_loss))
        
    
    print("Validation Loss : ", val_epoch_loss / 20)