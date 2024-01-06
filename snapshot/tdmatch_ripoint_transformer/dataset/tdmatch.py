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
        
        print()
        print("src_pcd : ", src_pcd[:10])
        print("tgt_pcd : ", tgt_pcd[:10])
        print("src_normals : ", src_normals[:10])
        print("tgt_normals : ", tgt_normals[:10])
        print("src_feats : ", src_feats[:10])
        print("tgt_feats : ", tgt_feats[:10])
        print("rot : ", rot[:10])
        print("trans : ", trans[:10])
        print()
        
        return src_pcd.astype(np.float32), tgt_pcd.astype(np.float32), \
               src_normals.astype(np.float32), tgt_normals.astype(np.float32),\
               src_feats.astype(np.float32), tgt_feats.astype(np.float32),\
               rot.astype(np.float32), trans.astype(np.float32),\
               src_pcd.astype(np.float32), None

    def __len__(self):
        return len(self.infos['rot'])


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
        self.aug_obj = aug.Augmentator([aug.Scaling([0.85, 1.15]), aug.Rotation([-180,180], 'fixed'), aug.Translation([-0.2, 0.2])])
        # self.aug_obj = aug.Augmentator([aug.Scaling([0.85, 1.15]), aug.Rotation([-180,180], 'rand'), aug.Translation([-0.2, 0.2])])

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
        
    
    def __len__(self):
        return len(self.stl_path_ls)
    
        
    def __getitem__(self, index):
        
        base_name = os.path.basename(self.stl_path_ls[index]).split(".")[0]
        loaded_json = gu.load_json(self.json_path_map[base_name])
        labels = torch.tensor(loaded_json['labels']).reshape(-1,1)
        if loaded_json['jaw'] == 'lower':
            labels -= 20
        labels[labels//10==1] %= 10
        labels[labels//10==2] = (labels[labels//10==2]%10) + 8
        labels[labels<0] = 0
        
        labels -= 1
        
        vertices, org_mesh = gu.read_txt_obj_ls(self.stl_path_ls[index], ret_mesh=True, use_tri_mesh=False)
        
        
        ''' GT check '''
        # print("labels :", labels.shape)
        # gu.print_3d(gu.get_colored_mesh(org_mesh, np.squeeze(labels+1, -1).detach().cpu().numpy()))
        '''OK'''
        
        
        labeled_vertices = np.concatenate([vertices, labels], 1)
        
        if labeled_vertices.shape[0]>24000:
            labeled_vertices = gu.resample_pcd([labeled_vertices], 24000, "fps")[0]
            
        labeled_vertices[:, :3] -= np.mean(labeled_vertices[:, :3], axis=0)
        # labeled_vertices[:, :3] = labeled_vertices[:, :3] / np.max(np.abs(labeled_vertices[:, :3]))
        # labeled_vertices[:, :3] = ((labeled_vertices[:, :3]-self.Y_AXIS_MIN)/(self.Y_AXIS_MAX - self.Y_AXIS_MIN))*2-1
        
        src_pcd = labeled_vertices[:, :6]
        label = labeled_vertices[:, -1:]
        
        src_normals = normal_redirect(src_pcd[:, :3], src_pcd[:, 3:], view_point=np.array([0., 0., 0.]))
        
        
        '''Augmentation'''
        if self.augmentation:
            self.aug_obj.reload_vals()
            src_pcd = self.aug_obj.run(src_pcd)
            
        
        src_pcd = torch.tensor(src_pcd)
        src_normals = torch.tensor(src_normals)
        src_feats = torch.ones(size=(src_pcd.shape[0], 1))
        label = torch.tensor(label)
        
        return src_pcd[:, :3].cuda().type(torch.float32), src_normals.cuda().type(torch.float32), src_feats.cuda().type(torch.float32), src_pcd[:, :3].cuda().type(torch.float32), label.cuda()
               