import torch.utils.data as data
import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from dataset.common import collect_local_neighbors, build_ppf_patch, farthest_point_subsampling,\
    point2node_sampling, calc_patch_overlap_ratio, get_square_distance_matrix, calc_ppf_cpu, sample_gt_node_correspondence, calc_gt_patch_correspondence, normal_redirect
from lib.utils import to_o3d_pcd
import open3d as o3d

from glob import glob
import gen_utils as gu

import augmentator as aug
import pyvista as pv
from time import time
import pymeshlab

from pytorch3d.transforms import so3_log_map, so3_exponential_map
import trimesh


class TDMatchDataset(data.Dataset):
    '''
    Load subsampled coordinates, relative rotation and translation
    Output (torch.Tensor):
    src_pcd: (N, 3) source point cloud
    tgt_pcd: (M, 3) target point cloud
    src_node_xyz: (n, 3) nodes sparsely sampled from source point cloud
    tgt_node_xyz: (m, 3) nodes sparsely sampled from target point cloud
    rot: (3, 3)
    trans: (3, 1)
    correspondences: (?, 3)
    '''

    def __init__(self, infos, config, data_augmentation=True):
        super(TDMatchDataset, self).__init__()
        # information of data
        self.infos = infos
        # root dir
        self.base_dir = config.root
        # whether to do data augmentation
        self.data_augmentation = data_augmentation
        #self.data_augmentation = True
        # configurations
        self.config = config
        # factor used to control the maximum rotation during data augmentation
        self.rot_factor = 1.
        # maximum noise used in data augmentation
        self.augment_noise = config.augment_noise
        # the maximum number allowed in each single frame of point cloud
        self.points_lim = 30000
        # can be in ['train', 'val', 'test']
        self.mode = config.mode
        # original benchmark or rotated benchmark
        self.rotated = config.rotated
        # view point
        self.view_point = np.array([0., 0., 0.])


    def __getitem__(self, index):

        # get gt transformation
        rot = self.infos['rot'][index]
        trans = self.infos['trans'][index]
        # get original input point clouds
        src_path = os.path.join(self.base_dir, self.infos['src'][index])
        tgt_path = os.path.join(self.base_dir, self.infos['tgt'][index])
        # remove a dirty data
        if src_path.split('/')[-2] == '7-scenes-fire' and src_path.split('/')[-1] == 'cloud_bin_19.pth':
            index = (index + 1) % self.__len__()
            rot = self.infos['rot'][index]
            trans = self.infos['trans'][index]
            # get original input point clouds
            src_path = os.path.join(self.base_dir, self.infos['src'][index])
            tgt_path = os.path.join(self.base_dir, self.infos['tgt'][index])

        src_pcd = torch.load(src_path)
        tgt_pcd = torch.load(tgt_path)

        ##################################################################################################
        # if we get too many points, we do random down-sampling
        if src_pcd.shape[0] > self.points_lim:
            idx = np.random.permutation(src_pcd.shape[0])[:self.points_lim]
            src_pcd = src_pcd[idx]

        if tgt_pcd.shape[0] > self.points_lim:
            idx = np.random.permutation(tgt_pcd.shape[0])[:self.points_lim]
            tgt_pcd = tgt_pcd[idx]

        ##################################################################################################
        # whether to augment data for training / to rotate data for testing
        if self.data_augmentation:
            # rotate the point cloud
            euler_ab = np.random.rand(3) * np.pi * 2. / self.rot_factor  # anglez, angley, anglex
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            if (np.random.rand(1)[0] > 0.5):
                src_pcd = np.matmul(rot_ab, src_pcd.T).T

                rot = np.matmul(rot, rot_ab.T)
            else:
                tgt_pcd = np.matmul(rot_ab, tgt_pcd.T).T

                rot = np.matmul(rot_ab, rot)
                trans = np.matmul(rot_ab, trans)
            # add noise
            src_pcd += (np.random.rand(src_pcd.shape[0], 3) - 0.5) * self.augment_noise
            tgt_pcd += (np.random.rand(tgt_pcd.shape[0], 3) - 0.5) * self.augment_noise
        # wheter test on rotated benchmark
        elif self.rotated:
            # rotate the point cloud
            np.random.seed(index)
            euler_ab = np.random.rand(3) * np.pi * 2. / self.rot_factor  # anglez, angley, anglex
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            if (np.random.rand(1)[0] > 0.5):
                src_pcd = np.matmul(rot_ab, src_pcd.T).T

                rot = np.matmul(rot, rot_ab.T)
            else:
                tgt_pcd = np.matmul(rot_ab, tgt_pcd.T).T

                rot = np.matmul(rot_ab, rot)
                trans = np.matmul(rot_ab, trans)
        else:
            pass

        if (trans.ndim == 1):
            trans = trans[:, None]
        ##################################################################################################
        # Normal estimation
        o3d_src_pcd = to_o3d_pcd(src_pcd)
        o3d_tgt_pcd = to_o3d_pcd(tgt_pcd)
        o3d_src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
        src_normals = np.asarray(o3d_src_pcd.normals)
        src_normals = normal_redirect(src_pcd, src_normals, view_point=self.view_point)
        o3d_tgt_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
        tgt_normals = np.asarray(o3d_tgt_pcd.normals)
        tgt_normals = normal_redirect(tgt_pcd, tgt_normals, view_point=self.view_point)
        src_feats = np.ones(shape=(src_pcd.shape[0], 1))
        tgt_feats = np.ones(shape=(tgt_pcd.shape[0], 1))

        return src_pcd.astype(np.float32), tgt_pcd.astype(np.float32), \
               src_normals.astype(np.float32), tgt_normals.astype(np.float32),\
               src_feats.astype(np.float32), tgt_feats.astype(np.float32),\
               rot.astype(np.float32), trans.astype(np.float32),\
               src_pcd.astype(np.float32), None

    def __len__(self):
        return len(self.infos['rot'])



### 데이터 로딩시마다 샘플링하는 로더 (실제로 논문의 실험에서는 사용하지 않음)
class DentalMeshDataset(data.Dataset):
    def __init__(self, split_with_txt_path=None, augmentation=False):
        super(DentalMeshDataset, self).__init__()
        
        self.source_obj_path = "../datasets/3D_scans_per_patient_obj_files"
        self.source_json_path = "../datasets/ground-truth_labels_instances"
        
        if split_with_txt_path:
            self.split_base_name_ls = []
            f = open(split_with_txt_path, 'r')
            while True:
                line = f.readline()
                if not line: break
                self.split_base_name_ls.append(line.strip())
            f.close()

            # temp_ls = []
            # for i in range(len(self.mesh_paths)):
            #     p_id = os.path.basename(self.mesh_paths[i]).split("_")[0]
            #     if p_id in self.split_base_name_ls:
            #         temp_ls.append(self.mesh_paths[i])
            # self.mesh_paths = temp_ls
            
        self.augmentation = augmentation
        # self.aug_obj = aug.Augmentator([aug.Scaling([0.85, 1.15]), aug.Rotation([-30,30], 'fixed'), aug.Translation([-0.2, 0.2])])
        # self.aug_obj = aug.Augmentator([aug.Scaling([0.85, 1.15]), aug.Rotation([-180,180], 'fixed'), aug.Translation([-0.2, 0.2])])
        # self.aug_obj = aug.Augmentator([aug.Scaling([0.85, 1.15]), aug.Rotation([-180,180], 'rand'), aug.Translation([-0.2, 0.2])])
        # self.aug_obj = aug.Augmentator([aug.Rotation([-30,30], 'fixed')])
        self.aug_obj = aug.Augmentator([aug.Rotation([-180,180], 'rand')])

        self.stl_path_ls = []
        # for dir_path in [
        #     x[0] for x in os.walk(self.source_obj_path)
        #     ][1:]:
        #     self.stl_path_ls += glob(os.path.join(dir_path,"*.obj"))
        for case in self.split_base_name_ls:
            self.stl_path_ls.append(os.path.join(self.source_obj_path, case, case+'_upper.obj'))
            self.stl_path_ls.append(os.path.join(self.source_obj_path, case, case+'_lower.obj'))
            
        self.json_path_map = {}
        # for dir_path in [
        #     x[0] for x in os.walk(self.source_json_path)
        #     ][1:]:
        #     for json_path in glob(os.path.join(dir_path,"*.json")):
        #         self.json_path_map[os.path.basename(json_path).split(".")[0]] = json_path
        for case in self.split_base_name_ls:
            self.json_path_map[case+'_upper'] = os.path.join(self.source_json_path, case, case+'_upper.json')
            self.json_path_map[case+'_lower'] = os.path.join(self.source_json_path, case, case+'_lower.json')
    
        self.view_point = np.array([0., 0., 0.])
        
        self.Y_AXIS_MAX = 33.15232091532151
        self.Y_AXIS_MIN = -36.9843781139949
        
        self.palet = np.array([
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
        
    
    def __len__(self):
        return len(self.stl_path_ls)
    
        
    def __getitem__(self, index):
        
        """Preprocessing"""
        
        base_name = os.path.basename(self.stl_path_ls[index]).split(".")[0]
        loaded_json = gu.load_json(self.json_path_map[base_name])
        labels = torch.tensor(loaded_json['labels']).reshape(-1)
        if loaded_json['jaw'] == 'lower':
            labels -= 20
        labels[labels//10==1] %= 10
        labels[labels//10==2] = (labels[labels//10==2]%10) + 8
        labels[labels<0] = 0
        
        vertices, org_mesh = gu.read_txt_obj_ls(self.stl_path_ls[index], ret_mesh=True, use_tri_mesh=False)
        
        ''' GT check '''
        # print("labels :", labels.shape)
        # gu.print_3d(gu.get_colored_mesh(org_mesh, np.squeeze(labels+1, -1).detach().cpu().numpy()))
        '''OK'''
        
        
        '''Previous Sampling - FPS'''
        # labeled_vertices = np.concatenate([vertices, np.expand_dims(labels, -1)], 1)
        
        # if labeled_vertices.shape[0]>24000:
        #     labeled_vertices = gu.resample_pcd([labeled_vertices], 24000, "fps")[0]
        ''''''
        
        
        '''Sampling #0 - Poisson Disk Sampling'''
        # colored_mesh = gu.get_colored_mesh(org_mesh, np.array(labels))
        
        # pcd = colored_mesh.sample_points_poisson_disk(24000)
        # vertices = np.array(pcd.points)
        # normals = np.array(pcd.normals)
        # color_ind = np.array(pcd.colors)
        # sampled_label = np.zeros(24000)
        # for i, p in enumerate(self.palet):
        #     sampled_label[((np.isclose(color_ind, p)).sum(-1)//3).astype(np.bool_)] = i
        # sampled_label -= 1
            
        # labeled_vertices = np.concatenate([vertices, normals, np.expand_dims(sampled_label, axis=-1).astype(np.int64)], 1)
        ''''''
        
        '''Sampling #1 - Point Cloud Simplification'''
        label_colors = np.zeros(vertices[:, :3].shape)

        for idx, p in enumerate(self.palet):
            label_colors[labels==idx] = self.palet[idx]
            
        label_colors = np.concatenate([label_colors, np.ones((label_colors.shape[0], 1))], axis=-1)
        
        colored_mesh = pymeshlab.Mesh(vertex_matrix = vertices[:, :3],
                                      v_normals_matrix = vertices[:, 3:6],
                                      v_color_matrix = label_colors)
        
        colored_mesh_set = pymeshlab.MeshSet()
        colored_mesh_set.add_mesh(colored_mesh)
        
        colored_mesh_set.generate_simplified_point_cloud(radius=pymeshlab.Percentage(0.3), exactnumflag=True)
        
        sampled_label = np.zeros(colored_mesh_set[1].vertex_color_matrix().shape[0])
        for i, p in enumerate(self.palet):
            sampled_label[((np.isclose(colored_mesh_set[1].vertex_color_matrix()[:,:3], p)).sum(-1)//3).astype(np.bool_)] = i
        
        labeled_vertices = np.concatenate([colored_mesh_set[1].vertex_matrix(), colored_mesh_set[1].vertex_normal_matrix(), np.expand_dims(sampled_label, axis=-1).astype(np.int64)], axis=1)
        ''''''
    
        labeled_vertices[:, :3] -= np.mean(labeled_vertices[:, :3], axis=0)
        # labeled_vertices[:, :3] = labeled_vertices[:, :3] / np.max(np.abs(labeled_vertices[:, :3]))
        # labeled_vertices[:, :3] = ((labeled_vertices[:, :3]-self.Y_AXIS_MIN)/(self.Y_AXIS_MAX - self.Y_AXIS_MIN))*2-1
        
        src_pcd_nrm = labeled_vertices[:, :6]
        
        '''Augmentation'''
        if self.augmentation:
            self.aug_obj.reload_vals()
            src_pcd_nrm, _ = self.aug_obj.run(src_pcd_nrm)
            
        src_pcd = src_pcd_nrm[:, :3]
        src_normals = src_pcd_nrm[:, 3:6]
        label = labeled_vertices[:, -1:]
        
        
        # src_normals = normal_redirect(src_pcd, src_normals, view_point=self.view_point)
        
        
        src_pcd = torch.tensor(src_pcd)
        src_normals = torch.tensor(src_normals)
        src_feats = torch.ones(size=(src_pcd.shape[0], 1))
        label = torch.tensor(label)
        
        return src_pcd.cuda().type(torch.float32), src_normals.cuda().type(torch.float32), src_feats.cuda().type(torch.float32), src_pcd.cuda().type(torch.float32), label.cuda()
    
    
    
### 미리 샘플링된 .npy 파일을 로드하는 로더
class DentalMeshSampledDataset(data.Dataset):
    def __init__(self, split_with_txt_path=None, augmentation=False):
        super(DentalMeshSampledDataset, self).__init__()
        
        
        ### Train/Test dataset을 분리하기 위한 .txt 파일 읽음
        if split_with_txt_path:
            self.split_base_name_ls = []
            f = open(split_with_txt_path, 'r')
            while True:
                line = f.readline()
                if not line: break
                self.split_base_name_ls.append(line.strip())
            f.close()

            # temp_ls = []
            # for i in range(len(self.mesh_paths)):
            #     p_id = os.path.basename(self.mesh_paths[i]).split("_")[0]
            #     if p_id in self.split_base_name_ls:
            #         temp_ls.append(self.mesh_paths[i])
            # self.mesh_paths = temp_ls
            
        
        ### Augmentator Rotation([랜덤 각도 범위], 회전 축), 회전축 ('fixed': Z축에 대해서만 랜덤한 범위로 회전, 'rand': X, Y, Z 모든 축에 대해서 랜덤한 범위로 회전)
        ### p.24 참고
        self.augmentation = augmentation
        self.aug_obj = aug.Augmentator([aug.Scaling([0.85, 1.15]), aug.Rotation([-30,30], 'fixed'), aug.Translation([-0.2, 0.2])])
        # self.aug_obj = aug.Augmentator([aug.Scaling([0.85, 1.15]), aug.Rotation([-180,180], 'fixed'), aug.Translation([-0.2, 0.2])])
        # self.aug_obj = aug.Augmentator([aug.Scaling([0.85, 1.15]), aug.Rotation([-180,180], 'rand'), aug.Translation([-0.2, 0.2])])
        
        
        ### 샘플링 결과 .npy 파일 저장 디렉터리
        self.source_path = './preprocessed_data_simplification'
        # self.source_path = './preprocessed_data_simplification_osstem'
        self.split_path_ls = []
        for case in self.split_base_name_ls:
            if os.path.exists(os.path.join(self.source_path, case+'_lower.npy')):
                self.split_path_ls.append(os.path.join(self.source_path, case+'_lower.npy'))
            if os.path.exists(os.path.join(self.source_path, case+'_upper.npy')):
                self.split_path_ls.append(os.path.join(self.source_path, case+'_upper.npy'))
        
        
        ### vertices 좌표 일반화 범위 (ToothGroupNet에서 사용한 숫자를 그대로 사용)
        self.Y_AXIS_MAX = 33.15232091532151
        self.Y_AXIS_MIN = -36.9843781139949
        
        
        ### Normal Redirection을 위한 기준점, normal 벡터들의 방향이 [0., 0., 0.]을 향하게 함
        ### (로딩시 vertices의 평균(중점)을 원점으로 옮기기 때문에, 결국은 normal들의 방향이 스캔 모델의 중심을 향하게 됨)
        ### p.26 참고
        self.view_point = np.array([0., 0., 0.])
        
    
    def __len__(self):
        return len(self.split_path_ls)
    
        
    def __getitem__(self, index):
        
        labeled_vertices = np.load(self.split_path_ls[index])
        
        labeled_vertices[:, :3] -= np.mean(labeled_vertices[:, :3], axis=0)        ## vertices 좌표들의 중심(중점)을 원점으로 옮김, 일종의 일반화
        # labeled_vertices[:, :3] = labeled_vertices[:, :3] / np.max(np.abs(labeled_vertices[:, :3]))
        labeled_vertices[:, :3] = ((labeled_vertices[:, :3]-self.Y_AXIS_MIN)/(self.Y_AXIS_MAX - self.Y_AXIS_MIN))*2-1       ## vertices 좌표들을 특정 범위 내로 일반화
        
        src_pcd_nrm = labeled_vertices[:, :6]
        
        '''Augmentation'''
        ### Augmentation 진행 (vertices, vertex normals에 똑같이 적용)
        if self.augmentation:
            self.aug_obj.reload_vals()
            src_pcd_nrm, _ = self.aug_obj.run(src_pcd_nrm)
            
        src_pcd = src_pcd_nrm[:, :3]        # Vertices
        src_normals = src_pcd_nrm[:, 3:6]   # Vertex Normals
        label = labeled_vertices[:, -1:]    # Label
        
        ### Normal Redirection을 위한 기준점, normal 벡터들의 방향이 [0., 0., 0.]을 향하게 함
        ### (로딩시 vertices의 평균(중점)을 원점으로 옮기기 때문에, 결국은 normal들의 방향이 스캔 모델의 중심을 향하게 됨)
        ### p.26 참고
        # src_normals = normal_redirect(src_pcd, src_normals, view_point=self.view_point)
        
        
        src_pcd = torch.tensor(src_pcd)                         # Vertices
        src_normals = torch.tensor(src_normals)                 # Vertex Normals
        src_feats = torch.ones(size=(src_pcd.shape[0], 1))      # Initial Features
        label = torch.tensor(label)                             # Label
        
        return src_pcd.cuda().type(torch.float32), src_normals.cuda().type(torch.float32), src_feats.cuda().type(torch.float32), src_pcd.cuda().type(torch.float32), label.cuda()




'''밑에부터는 Axes Regression에 사용했던 Data Loader'''



class DentalMeshDatasetAxes(data.Dataset):
    def __init__(self, mode, augmentation=False):
        super(DentalMeshDatasetAxes, self).__init__()
        
        self.source_path = '../datasets/osttemorigin'
        self.mode = mode
        
        ## Train / Val, Test Split
        if mode=='train':
            self.cases=  ['Case_11', 'Case_12', 'Case_13', 'Case_14', 'Case_15', 'Case_16', 'Case_17', 'Case_18', 'Case_19', 'Case_20',
                          'Case_21', 'Case_22', 'Case_23', 'Case_24', 'Case_25', 'Case_26', 'Case_27', 'Case_28', 'Case_29', 'Case_30',
                          'Case_31', 'Case_32', 'Case_33', 'Case_34', 'Case_35', 'Case_36', 'Case_37', 'Case_38', 'Case_39', 'Case_40',
                          'Case_41', 'Case_42', 'Case_43', 'Case_44', 'Case_45', 'Case_46']
        elif mode=='val' or mode=='test':
            self.cases = ['Case_01', 'Case_02', 'Case_03', 'Case_04', 'Case_05', 'Case_06', 'Case_07', 'Case_08', 'Case_09', 'Case_10']
        
        self.augmentation = augmentation
        
        ### Augmentator Rotation([랜덤 각도 범위], 회전 축), 회전축 ('fixed': Z축에 대해서만 랜덤한 범위로 회전, 'rand': X, Y, Z 모든 축에 대해서 랜덤한 범위로 회전)
        # self.aug_obj = aug.Augmentator([aug.Scaling([0.85, 1.15]), aug.Rotation([-30,30], 'fixed')])
        # self.aug_obj = aug.Augmentator([aug.Scaling([0.85, 1.15]), aug.Rotation([-30,30], 'fixed'), aug.Translation([-0.2, 0.2])])
        # self.aug_obj = aug.Augmentator([aug.Scaling([0.85, 1.15]), aug.Rotation([-180,180], 'fixed'), aug.Translation([-0.2, 0.2])])
        self.aug_obj = aug.Augmentator([aug.Scaling([0.85, 1.15]), aug.Rotation([-30,30], 'rand'), aug.Translation([-0.2, 0.2])])

        self.stl_path_ls = []
        for case in self.cases:
            file_list = os.listdir(os.path.join(self.source_path, case, 'STL'))
            self.stl_path_ls += [os.path.join(self.source_path, case, 'STL', file) for file in file_list if file.endswith('.stl')]
            
        ## Upper/Lower 축 상태가 비슷한 케이스끼리 그룹으로 묶음
        self.group1 = ['Case_01', 'Case_03', 'Case_04', 'Case_11', 'Case_13', 'Case_22', 'Case_23', 'Case_24', 'Case_25', 'Case_27', 'Case_37', 'Case_39', 'Case_40', 'Case_43', 'Case_46']
        self.group2 = ['Case_02', 'Case_05', 'Case_06', 'Case_07', 'Case_08', 'Case_09', 'Case_10', 'Case_12', 'Case_14', 'Case_15', 'Case_16', 'Case_17', 'Case_18', 'Case_19', 'Case_20',
                       'Case_21', 'Case_26', 'Case_28', 'Case_29', 'Case_30', 'Case_31', 'Case_32', 'Case_33', 'Case_34', 'Case_35', 'Case_36', 'Case_38', 'Case_41', 'Case_42', 'Case_44', 'Case_45']
        
        ## gt_mat1 : group1 lower & group2 upper 에 대한 GT Matrix
        ## gt_mat2 : group2 lower & group1 upper 에 대한 GT Matrix
        self.gt_mat1 = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
        self.gt_mat2 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        
    
    def __len__(self):
        return len(self.stl_path_ls)
    
        
    def __getitem__(self, index):
        case = self.stl_path_ls[index].split('/')[-3]
        if 'lower' in self.stl_path_ls[index]:
            jaw = 'lower'
        else:
            jaw = 'upper'
        
        """Preprocessing"""
        # vertices, org_mesh = gu.read_txt_obj_ls(self.stl_path_ls[index], ret_mesh=True, use_tri_mesh=False)
        loaded_mesh = trimesh.load_mesh(self.stl_path_ls[index])
        vertices = loaded_mesh.vertices
        normals = loaded_mesh.vertex_normals
        
        vertices -= np.mean(vertices, axis=0)       # 모델의 중점을 [0., 0., 0.] 으로 옮김
        # labeled_vertices[:, :3] = labeled_vertices[:, :3] / np.max(np.abs(labeled_vertices[:, :3]))
        # labeled_vertices[:, :3] = ((labeled_vertices[:, :3]-self.Y_AXIS_MIN)/(self.Y_AXIS_MAX - self.Y_AXIS_MIN))*2-1     # Vertex coordinate normalization
        
        """"""
        
        src_pcd = np.concatenate([vertices, normals], axis=-1)      
        
        '''Augmentation'''
        if self.augmentation:
            self.aug_obj.reload_vals()
            src_pcd, aug_mats = self.aug_obj.run(src_pcd)
            
        src_normals = src_pcd[:, 3:6]
        
        if ((case in self.group1) & (jaw=='lower')) | ((case in self.group2) & (jaw=='upper')):
            gt_mat = self.gt_mat1
        else:
            gt_mat = self.gt_mat2
        
        
        ### Augmentation 이후 gt_mat 만드는 과정 p.37~p.41 참고
        label_mat = np.matmul(np.linalg.inv(aug_mats[1]).T, gt_mat)
        
        # loaded_mesh.vertices = np.matmul(src_pcd[:, :3], label_mat)
        # loaded_mesh.vertex_normals = np.matmul(src_normals, label_mat)
        # loaded_mesh.export(self.stl_path_ls[index].split('/')[-1])
        
        label = so3_log_map(torch.tensor([label_mat]))[0] / np.pi
        ###
        
        
        src_pcd = torch.tensor(src_pcd)                         # Vertices
        src_normals = torch.tensor(src_normals)                 # Vertex Normals
        src_feats = torch.ones(size=(src_pcd.shape[0], 1))      # Initial Features
        label = torch.tensor(label)                             # Label
        
        
        return src_pcd[:, :3].cuda().type(torch.float32), src_normals.cuda().type(torch.float32), src_feats.cuda().type(torch.float32), src_pcd[:, :3].cuda().type(torch.float32), label.cuda().type(torch.float32)