import torch
from model.backbone_only import RIPointTransformer
import gen_utils as gu
from dataset.common import normal_redirect
import numpy as np
import pyvista as pv
import open3d as o3d
import pymeshlab
import os
import trimesh


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

base_dir = '../datasets/osttemorigin_annotated'
cases = sorted(os.listdir(base_dir))

# save_path = 'osstem_results_' + checkpoint_path.split('/')[1]
# save_path = 'osstem_aligned_results_' + checkpoint_path.split('/')[1]
save_path = 'osstem_results_aligned_' + checkpoint_path.split('/')[1]

if not os.path.exists(save_path):
    os.mkdir(save_path)

## Upper/Lower 축 상태가 비슷한 케이스끼리 그룹으로 묶음
group1 = ['Case_01', 'Case_03', 'Case_04', 'Case_11', 'Case_13', 'Case_22', 'Case_23', 'Case_24', 'Case_25', 'Case_27', 'Case_37', 'Case_39', 'Case_40', 'Case_43', 'Case_46']
group2 = ['Case_02', 'Case_05', 'Case_06', 'Case_07', 'Case_08', 'Case_09', 'Case_10', 'Case_12', 'Case_14', 'Case_15', 'Case_16', 'Case_17', 'Case_18', 'Case_19', 'Case_20',
          'Case_21', 'Case_26', 'Case_28', 'Case_29', 'Case_30', 'Case_31', 'Case_32', 'Case_33', 'Case_34', 'Case_35', 'Case_36', 'Case_38', 'Case_41', 'Case_42', 'Case_44', 'Case_45']

## gt_mat1 : group1 lower & group2 upper 에 대한 GT Matrix
## gt_mat2 : group2 lower & group1 upper 에 대한 GT Matrix
gt_mat1, gt_mat2 = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]), np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

for case in cases:
    for jaw in ['upper', 'lower']:
        mesh_path, gt_path = os.path.join(base_dir, case, 'STL', case+'_'+jaw+'.obj'), os.path.join(base_dir, case, 'STL', case+'_'+jaw+'.txt')

        vertices, org_mesh = gu.read_txt_obj_ls(mesh_path, ret_mesh=True, use_tri_mesh=True)

        src_pcd = vertices[:, :6]
        
        if ((case in group1) & (jaw=='lower')) | ((case in group2) & (jaw=='upper')):
            gt_mat = gt_mat1
        else:
            gt_mat = gt_mat2


        '''GT'''

        with open(gt_path, 'r') as txt_file:
                labels = txt_file.readlines()
                if len(labels)==2:
                    teeth = list(map(int, labels[-1].split()))
                else:
                    teeth = []

        with open(mesh_path, 'r') as obj_file:
            g, g_id = False, -1
            group_dict, group = {}, set()
            
            for line in obj_file:
                ### faces annotation read
                if g and line.startswith('f'):
                    group.update(set(map(int, set(line[2:].replace('//', ' ').split()))))
                    # continue
                ###

                ### tooth number annotation read
                elif g and line.startswith('#'):        
                    g = False
                    group_dict[teeth[g_id]] = sorted(list(group))
                    group = set()
                elif line.startswith('g') and line[9]!='0':
                    g = True
                    g_id+=1
                ###

            if teeth != []:
                group_dict[teeth[g_id]] = sorted(list(group))
                del(group)
                # print(group_dict.keys())
                
        gt_labels = np.zeros(len(src_pcd))
        # print("len :", len(gt_labels))
        for d in group_dict.keys():
            gt_labels[np.array(group_dict[d])-1] = int(d)
            
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
        
        points = np.array(mesh.points) - np.mean(mesh.points, axis=0)

        points = np.matmul(points, gt_mat)


        org_mesh.vertices = o3d.utility.Vector3dVector(points)
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


        cls_output, mask_output, sem_output = model([src_raw_pcd, src_feats, src_o, src_normals])
        cls_output = cls_output.argmax(-1)
        mask_output = mask_output.argmax(-1)
        
        mask_labels = mask_labels.reshape(-1)
        gt_labels = gt_labels.reshape(-1)
        lr_labels = lr_labels.reshape(-1)



        # print("Mask acc : {:.4f}".format((mask_output==torch.tensor(mask_labels, device='cuda')).sum() / mask_output.shape[0]))
        # print("Class acc : {:.4f}".format((cls_output==torch.tensor(gt_labels, device='cuda')).sum() / cls_output.shape[0]))

        ### 치아-잇몸 바이너리 클래스 예측에 대한 결과 가시화
        mask_pred_colored_mesh = gu.get_colored_mesh(org_mesh, mask_output.detach().cpu().numpy())
        # # print("Mask acc : {:.4f}".format((mask_output==torch.tensor(mask_labels, device='cuda')).sum() / mask_output.shape[0]))
        # mask_points = o3d.geometry.PointCloud()
        # mask_points.points = mask_pred_colored_mesh.vertices
        # mask_points.normals = mask_pred_colored_mesh.vertex_normals
        # mask_points.colors = mask_pred_colored_mesh.vertex_colors
        # o3d.visualization.draw_geometries([mask_points])
        # gu.print_3d(mask_pred_colored_mesh)
        ###
        
        
        
        ### 정합을 위해 tooth crown 부분만 (.obj) 파일로 저장했던 코드
        tri_mask_mesh = trimesh.Trimesh(vertices=mask_pred_colored_mesh.vertices, faces=mask_pred_colored_mesh.triangles)
        tri_mask_mesh.export(os.path.join(save_path, case+'_'+jaw+'.obj'))
        ###
        
        
        ### 치아 전체 클래스 예측에 대한 결과 가시화
        # cls_pred_colored_mesh = gu.get_colored_mesh(org_mesh, cls_output.detach().cpu().numpy())
        # # # print("Class acc : {:.4f}".format((cls_output==torch.tensor(gt_labels, device='cuda')).sum() / cls_output.shape[0]))
        # # cls_points = o3d.geometry.PointCloud()
        # # cls_points.points = cls_pred_colored_mesh.vertices
        # # cls_points.normals = cls_pred_colored_mesh.vertex_normals
        # # cls_points.colors = cls_pred_colored_mesh.vertex_colors
        # # o3d.visualization.draw_geometries([cls_points])
        # gu.print_3d(cls_pred_colored_mesh)
        ###
        

        ### mesh to point clouds
        # pcl = o3d.geometry.PointCloud()
        # pcl.points = cls_pred_colored_mesh.vertices
        # pcl.colors = cls_pred_colored_mesh.vertex_colors
        # o3d.visualization.draw_geometries([pcl])
        ###

