import torch
from model.backbone_only import RIPointTransformer
import gen_utils as gu
from dataset.common import normal_redirect
import numpy as np

import pyvista as pv
import open3d as o3d

import pymeshlab


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

# checkpoint_path = 'checkpoints/lims_without_norm_5_encdec_multiscale_cls_mask+cls/epoch61_val0.5790_cls_acc0.7587_mask_acc0.9796.pth'
# checkpoint_path = 'checkpoints/lims_without_norm_5_encdec_multiscale_cls_mask+cls/epoch90_val0.8209_cls_acc0.7884_mask_acc0.9821.pth'
# checkpoint_path = 'checkpoints/lims_without_norm_5_encdec_multiscale_cls_mask+cls/epoch81_val0.6884_cls_acc0.7924_mask_acc0.9819.pth'
# checkpoint_path = 'checkpoints/lims_without_norm_5_encdec_multiscale_cls_mask+cls_sampled_saved/epoch98_val0.2551_cls_acc0.9070_mask_acc0.9914.pth'
# checkpoint_path = 'checkpoints/lims_without_norm_5_encdec_multiscale_cls_mask+cls_sampled_focal_1/epoch86_val0.1673_cls_acc0.9108_mask_acc0.9921.pth'
# checkpoint_path = 'checkpoints/lims_without_norm_5_encdec_multiscale_cls_mask+cls_simplification_focal/epoch83_val0.4873_cls_acc0.7289_mask_acc0.9834.pth'
# checkpoint_path = 'checkpoints/lims_without_norm_5_encdec_multiscale_cls_mask+cls_simplified_normredir/epoch84_val0.5947_cls_acc0.7669_mask_acc0.9821.pth'
# checkpoint_path = 'checkpoints/lims_without_norm_5_encdec_multiscale_cls_mask+cls_simplified_normredir/epoch99_val0.6429_cls_acc0.7617_mask_acc0.9834.pth'
# checkpoint_path = 'checkpoints/lims_without_norm_5_encdec_multiscale_cls_mask+cls_simplified_normredir/epoch84_val0.5947_cls_acc0.7669_mask_acc0.9821.pth'
# checkpoint_path = 'checkpoints/lims_without_norm_5_encdec_multiscale_cls_mask+cls_simplified_normredir_focal/epoch83_val0.5781_cls_acc0.7416_mask_acc0.9783.pth'
# checkpoint_path = 'checkpoints_rollback/simplification_saved_maskhead_multiscale_vertexnorm/epoch83_val0.9012_cls_acc0.8159_mask_acc0.9616.pth'
# checkpoint_path = 'checkpoints_osstem/simplification_saved_maskhead_multiscale_vertexnorm_LRhead_focal/epoch86_val1.0023_cls_acc0.8355_mask_acc0.9400_LR_acc0.8767.pth'
# checkpoint_path = 'checkpoints_osstem/simplification_saved_maskhead_multiscale_vertexnorm_focal_normredir/epoch81_val0.6513_cls_acc0.8297_mask_acc0.9242_.pth'


checkpoint_path = 'checkpoints_osstem/simplification_saved_maskhead_multiscale_vertexnorm/epoch72_val0.7411_cls_acc0.8342_mask_acc0.9571_.pth'
# checkpoint_path = 'checkpoints_rollback/simplification_saved_maskhead_multiscale_vertexnorm_normredirect/epoch82_val0.7044_cls_acc0.8453_mask_acc0.9674_.pth'


# mesh_path = 'test_input_aligned/Case_01_lower(Antagonist)_1.obj'
# mesh_path = 'test_input_aligned/Case_01_upper_1.obj'
# mesh_path = 'test_input_aligned/Case_03_lower(Antagonist)_1.obj'
# mesh_path, gt_path = '../datasets/osttemorigin_annotated/Case_40/STL/Case_40_lower.obj', '../datasets/osttemorigin_annotated/Case_40/STL/Case_40_lower.txt'
mesh_path, gt_path = '../datasets/osttemorigin_annotated/Case_40/STL/Case_40_upper.obj', '../datasets/osttemorigin_annotated/Case_40/STL/Case_40_upper.txt'


vertices, org_mesh = gu.read_txt_obj_ls(mesh_path, ret_mesh=True, use_tri_mesh=True)

src_pcd = vertices[:, :6]


'''GT'''

### GT reading
with open(gt_path, 'r') as txt_file:
        labels = txt_file.readlines()
        if len(labels)==2:
            teeth = list(map(int, labels[-1].split()))
        else:
            teeth = []
###


### OBJ reading
with open(mesh_path, 'r') as obj_file:
    g, g_id = False, -1
    group_dict, group = {}, set()
    
    for line in obj_file:
        if g and line.startswith('f'):
            group.update(set(map(int, set(line[2:].replace('//', ' ').split()))))
        elif g and line.startswith('#'):
            g = False
            group_dict[teeth[g_id]] = sorted(list(group))
            group = set()
        elif line.startswith('g') and line[9]!='0':
            g = True
            g_id+=1
    if teeth != []:
        group_dict[teeth[g_id]] = sorted(list(group))
        del(group)
###
        
        
gt_labels = np.zeros(len(src_pcd))
for d in group_dict.keys():
    gt_labels[np.array(group_dict[d])-1] = int(d)


### 치아 번호들을 0(잇몸), 1~8(#11~#18, #31~#38), 9~16(#21~#28, #41~48) 로 변경해줌
if 'lower' in mesh_path:
    gt_labels -= 20
gt_labels[gt_labels//10==1] %= 10
gt_labels[gt_labels//10==2] = (gt_labels[gt_labels//10==2]%10) + 8
gt_labels[gt_labels<0] = 0

''''''


'''Previous Sampling'''
# if src_pcd.shape[0] > 24000:
#     src_pcd = gu.resample_pcd([src_pcd], 24000, "fps")[0]



"""Sampling #0 - Poisson Disk Sampling"""
        
# pcd = org_mesh.sample_points_poisson_disk(24000)
# # o3d.visualization.draw_geometries([pcd])
# vertices = np.array(pcd.points)
# normals = np.array(pcd.normals)
    
# src_pcd = np.concatenate([vertices, normals], 1)
# """"""



"""Sampling #1 - Point Cloud Simplification"""
label_colors = np.zeros(vertices[:, :3].shape)

for idx, p in enumerate(palet):
    label_colors[gt_labels==idx] = palet[idx]
    
label_colors = np.concatenate([label_colors, np.ones((label_colors.shape[0], 1))], axis=-1)

meshlab_mesh = pymeshlab.Mesh(vertex_matrix = vertices[:, :3],
                              v_normals_matrix = vertices[:, 3:6],
                              v_color_matrix = label_colors)
colored_mesh_set = pymeshlab.MeshSet()
colored_mesh_set.add_mesh(meshlab_mesh)

colored_mesh_set.generate_simplified_point_cloud(radius=pymeshlab.Percentage(0.3), exactnumflag=True)

sampled_label = np.zeros(colored_mesh_set[1].vertex_color_matrix().shape[0])
for i, p in enumerate(palet):
    sampled_label[((np.isclose(colored_mesh_set[1].vertex_color_matrix()[:,:3], p)).sum(-1)//3).astype(np.bool_)] = i


labeled_vertices = np.concatenate([colored_mesh_set[1].vertex_matrix(), colored_mesh_set[1].vertex_normal_matrix(),
                                   np.expand_dims(sampled_label, axis=-1).astype(np.int64)], axis=1)
""""""


"""GT Check"""
gt_labels = labeled_vertices[:, -1:]

mask_labels = np.copy(gt_labels)
mask_labels[mask_labels>0] = 1

# gu.print_3d(gu.get_colored_mesh(org_mesh, gt_labels.reshape(-1)))
"""OK"""



'''Remeshing'''
cloud = pv.PolyData(labeled_vertices[:,:3])
mesh = cloud.delaunay_2d()

org_mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.points))
org_mesh.triangles = o3d.utility.Vector3iVector(np.array(mesh.regular_faces))
org_mesh.compute_vertex_normals()
''''''


labeled_vertices[:, :3] -= np.mean(labeled_vertices[:, :3], axis=0)
labeled_vertices[:, :3] = ((labeled_vertices[:, :3]-Y_AXIS_MIN)/(Y_AXIS_MAX - Y_AXIS_MIN))*2-1

src_pcd = labeled_vertices[:, :3]
src_normals = labeled_vertices[:, 3:6]

# src_normals = normal_redirect(src_pcd[:,:3], src_normals, view_point=view_point)


src_pcd, src_normals, src_feats, src_raw_pcd = torch.tensor(src_pcd)[:, :3].cuda().type(torch.float32).contiguous(), \
                                            torch.tensor(src_normals).cuda().type(torch.float32).contiguous(), \
                                            torch.ones(size=(src_pcd.shape[0], 1)).cuda().type(torch.float32).contiguous(), \
                                            torch.tensor(src_pcd)[:, :3].cuda().type(torch.float32).contiguous()
src_o = torch.tensor([src_raw_pcd.shape[0]]).to(src_raw_pcd).int().contiguous()


model = RIPointTransformer(transformer_architecture=['self', 'cross', 'self', 'cross', 'self', 'cross'], with_cross_pos_embed=True, factor=1)
model.cuda()
model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
model.eval()

# print(model)


cls_output, mask_output, sem_output = model([src_raw_pcd, src_feats, src_o, src_normals])
cls_output = cls_output.argmax(-1)
mask_output = mask_output.argmax(-1)

mask_labels = mask_labels.reshape(-1)
gt_labels = gt_labels.reshape(-1)
lr_labels = lr_labels.reshape(-1)



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


mask_pred_colored_mesh = gu.get_colored_mesh(org_mesh, mask_output.detach().cpu().numpy())
# # print("Mask acc : {:.4f}".format((mask_output==torch.tensor(mask_labels, device='cuda')).sum() / mask_output.shape[0]))
# mask_points = o3d.geometry.PointCloud()
# mask_points.points = mask_pred_colored_mesh.vertices
# mask_points.normals = mask_pred_colored_mesh.vertex_normals
# mask_points.colors = mask_pred_colored_mesh.vertex_colors
# o3d.visualization.draw_geometries([mask_points])
gu.print_3d(mask_pred_colored_mesh)

cls_pred_colored_mesh = gu.get_colored_mesh(org_mesh, cls_output.detach().cpu().numpy())
# # print("Class acc : {:.4f}".format((cls_output==torch.tensor(gt_labels, device='cuda')).sum() / cls_output.shape[0]))
# cls_points = o3d.geometry.PointCloud()
# cls_points.points = cls_pred_colored_mesh.vertices
# cls_points.normals = cls_pred_colored_mesh.vertex_normals
# cls_points.colors = cls_pred_colored_mesh.vertex_colors
# o3d.visualization.draw_geometries([cls_points])
gu.print_3d(cls_pred_colored_mesh)

### mesh to point clouds
# pcl = o3d.geometry.PointCloud()
# pcl.points = cls_pred_colored_mesh.vertices
# pcl.colors = cls_pred_colored_mesh.vertex_colors
# o3d.visualization.draw_geometries([pcl])
###