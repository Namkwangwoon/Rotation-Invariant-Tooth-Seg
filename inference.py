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


# palet = np.array([
#             [255,153,153],

#             [153,76,0],
#             [153,153,0],
#             [76,153,0],
#             [0,153,153],
#             [0,0,153],
#             [153,0,153],
#             [153,0,76],
#             [64,64,64],

#             [255,128,0],
#             [153,153,0],
#             [76,153,0],
#             [0,153,153],
#             [0,0,153],
#             [153,0,153],
#             [153,0,76],
#             [64,64,64],
#             ])/255
# palet[9:] *= 0.4


### Visualization시 잇몸 & 이빨에 칠할 색깔들의 RGB값
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
###


### Vertex nomalization 범위
Y_AXIS_MAX = 33.15232091532151
Y_AXIS_MIN = -36.9843781139949
###


view_point = np.array([0., 0., 0.])     ## normal_redirection()을 위한 점


aug_obj = aug.Augmentator([aug.Rotation([-180,180], 'rand')])       ## input을 회전시키기 위함


parser = argparse.ArgumentParser()
parser.add_argument('--model', default=None, type=str)      ## 아무것도 입력하지 않으면 제안한 모델, pointnet, pointtransformer 입력 가능
args = parser.parse_args()


'''Checkpoints path'''
checkpoint_path = 'checkpoints_rollback/simplification_saved_maskhead_multiscale_vertexnorm_normredirect/epoch82_val0.7044_cls_acc0.8453_mask_acc0.9674_.pth'
# checkpoint_path = 'checkpoints_rollback/simplification_saved_maskhead_multiscale_vertexnorm_normdir_LRhead_focal/epoch90_val1.0673_cls_acc0.8290_mask_acc0.9584_LR_acc0.8935.pth'
# checkpoint_path = 'checkpoints_rollback/simplification_saved_maskhead_multiscale_vertexnorm/epoch83_val0.9012_cls_acc0.8159_mask_acc0.9616.pth'
# checkpoint_path = 'checkpoints_rotate/vertexnorm/epoch66_val0.8429_cls_acc0.7969_mask_acc0.9573_.pth'

## PointNet
# checkpoint_path = 'checkpoints_pointnet/vertexnorm/epoch97_val1.1880_cls_acc0.7050_mask_acc0.0000.pth'
# checkpoint_path = 'checkpoints_pointnet/rotate_aligned_vertexnorm/epoch58_val2.2611_cls_acc0.5248_mask_acc0.0000.pth'

## Point Transformer
# checkpoint_path = 'checkpoints_pointtransformer/vertexnorm/epoch90_val1.2929_cls_acc0.8183_mask_acc0.0000.pth'
# checkpoint_path = 'checkpoints_pointtransformer/rotate_aligned_vertexnorm/epoch74_val2.9434_cls_acc0.5932_mask_acc0.0000.pth'
''''''


''' input으로 줄 case 번호 (../datasets/3D_scans_per_patient_obj_files의 목록 중 하나)
    inference이기 때문에, validation set 혹은 test set, 즉, 분리해준 "base_name_test_fold.txt" 의 목록 중 하나여야 함 '''
mesh_num = '015XE9MT'
# mesh_num = '01FJT0PR'
flag = 'lower'
flag = 'upper'


'''input mesh 및 gt의 path'''
mesh_dir = '../datasets/3D_scans_per_patient_obj_files'
gt_dir = '../datasets/ground-truth_labels_instances'
''''''


'''실제 .obj 파일 및 .json 파일의 위치'''
mesh_path = os.path.join(mesh_dir, mesh_num, mesh_num+'_'+flag+'.obj')
gt_path = os.path.join(gt_dir, mesh_num, mesh_num+'_'+flag+'.json')
''''''


vertices, org_mesh = gu.read_txt_obj_ls(mesh_path, ret_mesh=True, use_tri_mesh=False)

src_pcd = vertices[:, :6]



'''GT Loading'''
gt_loaded_json = gu.load_json(gt_path)
gt_labels = np.array(gt_loaded_json['labels']).reshape(-1)

if gt_loaded_json['jaw'] == 'lower':
    gt_labels -= 20
gt_labels[gt_labels//10==1] %= 10
gt_labels[gt_labels//10==2] = (gt_labels[gt_labels//10==2]%10) + 8
gt_labels[gt_labels<0] = 0
''''''



'''Previous Sampling - FPS'''
# if labeled_vertices.shape[0] > 24000:
#     labeled_vertices = gu.resample_pcd([labeled_vertices], 24000, "fps")[0]
''''''



'''Sampling #0 - Poisson Disk Sampling'''
# colored_mesh = gu.get_colored_mesh(org_mesh, gt_labels)
        
# pcd = colored_mesh.sample_points_poisson_disk(24000)
# vertices = np.array(pcd.points)
# normals = np.array(pcd.normals)
# color_ind = np.array(pcd.colors)
# sampled_label = np.zeros(24000)
# for i, p in enumerate(palet):
#     sampled_label[((np.isclose(color_ind, p)).sum(-1)//3).astype(np.bool_)] = i
    
# labeled_vertices = np.concatenate([vertices, normals, np.expand_dims(sampled_label, axis=-1).astype(np.int64)], 1)

''''''



'''Sampling #1 - Point Cloud Simplification'''
label_colors = np.zeros(vertices[:, :3].shape)

for idx, p in enumerate(palet):
    label_colors[gt_labels==idx] = palet[idx]
    
label_colors = np.concatenate([label_colors, np.ones((label_colors.shape[0], 1))], axis=-1)

colored_mesh = pymeshlab.Mesh(vertex_matrix = vertices[:, :3],
                                v_normals_matrix = vertices[:, 3:6],
                                v_color_matrix = label_colors)

colored_mesh_set = pymeshlab.MeshSet()
colored_mesh_set.add_mesh(colored_mesh)

colored_mesh_set.generate_simplified_point_cloud(radius=pymeshlab.Percentage(0.3), exactnumflag=True)

sampled_label = np.zeros(colored_mesh_set[1].vertex_color_matrix().shape[0])
for i, p in enumerate(palet):
    sampled_label[((np.isclose(colored_mesh_set[1].vertex_color_matrix()[:,:3], p)).sum(-1)//3).astype(np.bool_)] = i

labeled_vertices = np.concatenate([colored_mesh_set[1].vertex_matrix(), colored_mesh_set[1].vertex_normal_matrix(),
                                   np.expand_dims(sampled_label, axis=-1).astype(np.int64)], axis=1)



'''GT Check'''
gt_labels = labeled_vertices[:, -1:]


mask_labels = np.copy(gt_labels)
mask_labels[mask_labels>0] = 1

lr_labels = np.copy(gt_labels)
lr_labels[(lr_labels>=1) & (lr_labels<=8)] = 1
lr_labels[(lr_labels>=9) & (lr_labels<=16)] = 2

# gu.print_3d(gu.get_colored_mesh(org_mesh, gt_labels.reshape(-1)))     ## gt가 잘 로드되었는지 가시화 가능
'''OK'''



'''Remeshing'''
### 샘플링한 포인트 클라우드의 분할 결과를 가시화하기 위해서는 다시 Remeshing 해줘야함
### 기존 정점들이 없어져서 Face가 다 꼬여버림

cloud = pv.PolyData(labeled_vertices[:,:3])
mesh = cloud.delaunay_2d()

org_mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.points))
org_mesh.triangles = o3d.utility.Vector3iVector(np.array(mesh.regular_faces))
org_mesh.compute_vertex_normals()
''''''


labeled_vertices[:, :3] -= np.mean(labeled_vertices[:, :3], axis=0)
labeled_vertices[:, :3] = ((labeled_vertices[:, :3]-Y_AXIS_MIN)/(Y_AXIS_MAX - Y_AXIS_MIN))*2-1

src_pcd_nrm = labeled_vertices[:, :6]


'''Augmentation?!'''
## 입력 축을 랜덤으로 돌리고 싶으면 주석 풀기

# aug_obj.reload_vals()
# src_pcd_nrm, _ = aug_obj.run(src_pcd_nrm)
''''''


src_pcd = src_pcd_nrm[:, :3]
src_normals = src_pcd_nrm[:, 3:6]
label = labeled_vertices[:, -1:]

src_normals = normal_redirect(src_pcd, src_normals, view_point=view_point)      ## 학습 시(dataset/tdmatch.py) normal_redirect()를 사용해 학습했다면 주석 풀기


src_pcd, src_normals, src_feats, src_raw_pcd = torch.tensor(src_pcd).cuda().type(torch.float32).contiguous(), \
                                            torch.tensor(src_normals).cuda().type(torch.float32).contiguous(), \
                                            torch.ones(size=(src_pcd.shape[0], 1)).cuda().type(torch.float32).contiguous(), \
                                            torch.tensor(src_pcd).cuda().type(torch.float32).contiguous()
src_o = torch.tensor([src_raw_pcd.shape[0]]).to(src_raw_pcd).int().contiguous()



if args.model == None:
    model = RIPointTransformer(transformer_architecture=['self', 'cross', 'self', 'cross', 'self', 'cross'], with_cross_pos_embed=True, factor=1)
elif args.model == 'pointnet':
    model = get_model(17)
elif args.model == 'pointtransformer':
    model = PointTransformerSeg(PointTransformerBlock, [2, 3, 4, 6, 3], c=1, k=17)
model.cuda()
# model.load_state_dict(torch.load(checkpoint_path))
model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
model.eval()


if args.model == None:
    cls_output, mask_output, lr_output = model([src_raw_pcd, src_feats, src_o, src_normals])
    mask_output = mask_output.argmax(-1)
elif args.model == 'pointnet':
    src_raw_pcd = src_raw_pcd.unsqueeze(0).permute(0, 2, 1)
    cls_output = model(src_raw_pcd)[0]
elif args.model == 'pointtransformer':
    cls_output = model([src_raw_pcd, src_feats, src_o])
    
    
cls_output = cls_output.argmax(-1)

mask_labels = mask_labels.reshape(-1)
gt_labels = gt_labels.reshape(-1)
lr_labels = lr_labels.reshape(-1)


### Class별 정확도 측정 Class 0: 잇몸, Class 1~16: 11~18(or 31~38) + 21~28(or 41~48) 치아
### Acc: (클래스를 맞춘 포인트의 갯수 / 전체 포인트의 갯수)
if args.model == None:
    print("Mask acc : {:.4f}".format((mask_output==torch.tensor(mask_labels, device='cuda')).sum() / mask_output.shape[0]))
print("Class acc : {:.4f}".format((cls_output==torch.tensor(gt_labels, device='cuda')).sum() / cls_output.shape[0]))
print()
print("Class 0 acc : {:.4f}  ({}/{})".format((cls_output[gt_labels==0]==0).sum() / (gt_labels==0).sum(), (cls_output[gt_labels==0]==0).sum(), (gt_labels==0).sum()))
print("Class 1 acc : {:.4f}  ({}/{})".format((cls_output[gt_labels==1]==1).sum() / (gt_labels==1).sum(), (cls_output[gt_labels==1]==1).sum(), (gt_labels==1).sum()))
print("Class 2 acc : {:.4f}  ({}/{})".format((cls_output[gt_labels==2]==2).sum() / (gt_labels==2).sum(), (cls_output[gt_labels==2]==2).sum(), (gt_labels==2).sum()))
print("Class 3 acc : {:.4f}  ({}/{})".format((cls_output[gt_labels==3]==3).sum() / (gt_labels==3).sum(), (cls_output[gt_labels==3]==3).sum(), (gt_labels==3).sum()))
print("Class 4 acc : {:.4f}  ({}/{})".format((cls_output[gt_labels==4]==4).sum() / (gt_labels==4).sum(), (cls_output[gt_labels==4]==4).sum(), (gt_labels==4).sum()))
print("Class 5 acc : {:.4f}  ({}/{})".format((cls_output[gt_labels==5]==5).sum() / (gt_labels==5).sum(), (cls_output[gt_labels==5]==5).sum(), (gt_labels==5).sum()))
print("Class 6 acc : {:.4f}  ({}/{})".format((cls_output[gt_labels==6]==6).sum() / (gt_labels==6).sum(), (cls_output[gt_labels==6]==6).sum(), (gt_labels==6).sum()))
print("Class 7 acc : {:.4f}  ({}/{})".format((cls_output[gt_labels==7]==7).sum() / (gt_labels==7).sum(), (cls_output[gt_labels==7]==7).sum(), (gt_labels==7).sum()))
print("Class 8 acc : {:.4f}  ({}/{})".format((cls_output[gt_labels==8]==8).sum() / (gt_labels==8).sum(), (cls_output[gt_labels==8]==8).sum(), (gt_labels==8).sum()))
print("Class 9 acc : {:.4f}  ({}/{})".format((cls_output[gt_labels==9]==9).sum() / (gt_labels==9).sum(), (cls_output[gt_labels==9]==9).sum(), (gt_labels==9).sum()))
print("Class 10 acc : {:.4f}  ({}/{})".format((cls_output[gt_labels==10]==10).sum() / (gt_labels==10).sum(), (cls_output[gt_labels==10]==10).sum(), (gt_labels==10).sum()))
print("Class 11 acc : {:.4f}  ({}/{})".format((cls_output[gt_labels==11]==11).sum() / (gt_labels==11).sum(), (cls_output[gt_labels==11]==11).sum(), (gt_labels==11).sum()))
print("Class 12 acc : {:.4f}  ({}/{})".format((cls_output[gt_labels==12]==12).sum() / (gt_labels==12).sum(), (cls_output[gt_labels==12]==12).sum(), (gt_labels==12).sum()))
print("Class 13 acc : {:.4f}  ({}/{})".format((cls_output[gt_labels==13]==13).sum() / (gt_labels==13).sum(), (cls_output[gt_labels==13]==13).sum(), (gt_labels==13).sum()))
print("Class 14 acc : {:.4f}  ({}/{})".format((cls_output[gt_labels==14]==14).sum() / (gt_labels==14).sum(), (cls_output[gt_labels==14]==14).sum(), (gt_labels==14).sum()))
print("Class 15 acc : {:.4f}  ({}/{})".format((cls_output[gt_labels==15]==15).sum() / (gt_labels==15).sum(), (cls_output[gt_labels==15]==15).sum(), (gt_labels==15).sum()))
print("Class 16 acc : {:.4f}  ({}/{})".format((cls_output[gt_labels==16]==16).sum() / (gt_labels==16).sum(), (cls_output[gt_labels==16]==16).sum(), (gt_labels==16).sum()))
###


### 치아-잇몸 바이너리 클래스 예측에 대한 결과 가시화
# if args.model == None:
#     mask_pred_colored_mesh = gu.get_colored_mesh(org_mesh, mask_output.detach().cpu().numpy())
#     gu.print_3d(mask_pred_colored_mesh)
###


### 치아 전체 클래스 예측에 대한 결과 가시화
cls_pred_colored_mesh = gu.get_colored_mesh(org_mesh, cls_output.detach().cpu().numpy())        ## 예측 결과를 바탕으로 mesh에 color 부여
gu.print_3d(cls_pred_colored_mesh)                                                              ## opend3d 라이브러리를 사용한 가시화


### 또 다른 가시화 방법 (mesh to point clouds)
### point들 & 각 point들의 색깔에 대한 가시화 가능
pcl = o3d.geometry.PointCloud()
pcl.points = cls_pred_colored_mesh.vertices
pcl.colors = cls_pred_colored_mesh.vertex_colors
o3d.visualization.draw_geometries([pcl])
###


### 샘플링한 레이블에 대한 결과 가시화
# mask_gt_colored_mesh = gu.get_colored_mesh(org_mesh, torch.tensor(mask_labels, device='cuda').detach().cpu().numpy())
# gu.print_3d(mask_gt_colored_mesh)

# cls_gt_colored_mesh = gu.get_colored_mesh(org_mesh, torch.tensor(gt_labels, device='cuda').detach().cpu().numpy())
# gu.print_3d(cls_gt_colored_mesh)
###