import open3d as o3d
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import json
import pointops
import trimesh

import sys

def np_to_pcd(arr, color=[1,0,0]):
    arr = np.array(arr)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:,:3])
    if arr.shape[1] >= 6:
        pcd.normals = o3d.utility.Vector3dVector(arr[:,3:6])
    pcd.colors = o3d.utility.Vector3dVector([color]*len(pcd.points))
    return pcd

def save_pcd(path, arr):
    o3d.io.write_point_cloud(path, arr)

def save_mesh(path, mesh):
    o3d.io.write_triangle_mesh(path, mesh)

def count_unique_by_row(a):
    weight = 1j*np.linspace(0, a.shape[1], a.shape[0], endpoint=False)
    b = a + weight[:, np.newaxis]
    u, ind, cnt = np.unique(b, return_index=True, return_counts=True)
    b = np.zeros_like(a)
    np.put(b, ind, cnt)
    return b

def load_mesh(mesh_path, only_tooth_crop = False):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    
    if only_tooth_crop:
        cluster_idxes, cluster_nums, _ = mesh.cluster_connected_triangles()
        cluster_idxes = np.asarray(cluster_idxes)
        cluster_nums = np.asarray(cluster_nums)
        tooth_cluster_num = np.argmax(cluster_nums)
        mesh.remove_triangles_by_mask(cluster_idxes!=tooth_cluster_num)
    return mesh

def get_colored_mesh(mesh, label_arr):
    palte = np.array([
        [255,153,153],

        [153,76,0],
        [153,153,0],
        [76,153,0],
        [0,153,153],
        [0,0,153],
        [153,0,153],
        [153,0,76],
        [64,64,64],

        [255,128,0],
        [153,153,0],
        [76,153,0],
        [0,153,153],
        [0,0,153],
        [153,0,153],
        [153,0,76],
        [64,64,64],
    ])/255
    palte[9:] *= 0.4
    label_arr = label_arr.copy()
    label_arr %= palte.shape[0]
    label_colors = np.zeros((label_arr.shape[0], 3))
    for idx, palte_color in enumerate(palte):
        label_colors[label_arr==idx] = palte[idx]
        
    mesh.vertex_colors = o3d.utility.Vector3dVector(label_colors)
    
    
    '''제거 하고 싶은 클래스를 label_arr를 통해 지정해주면 해당 클래스가 제거된 mesh 생성 가능'''
    # remove_idx = (label_arr == 0)     ## label이 0인 클래스(잇몸)을 제거하기 위한 index
    # remain_idx = (label_arr != 0)     ## label이 0인 클래스(잇몸)을 남겨두기 위한 index
    
    # remove_num = np.arange(len(remove_idx))[remove_idx]
    # remove_num = np.flip(remove_num)
    # mesh_vertices = np.array(mesh.vertices)
    # mesh_tris = np.array(mesh.triangles)
    # print()
    # print("vertices :", mesh_vertices.shape)
    # mesh.vertices, mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_vertices[remain_idx]), o3d.utility.Vector3dVector(label_colors[remain_idx])
    # print(">>>")
    # print("vertices :", mesh_vertices[remain_idx].shape)
    # print()
    
    # mesh_tris, remove_num = torch.tensor(mesh_tris.copy()).cuda(), torch.tensor(remove_num.copy()).cuda()
    # tri = []
    # for t in mesh_tris:
    #     if t[0] in remove_num or t[1] in remove_num or t[2] in remove_num:
    #         continue
    #     tri.append(t)
    # tri = torch.stack(tri).cuda()
    # for num in remove_num:
    #     minus_ind = num < tri
    #     tri[minus_ind] -= 1
    # tri = tri.detach().cpu().numpy()
    # mesh.triangles = o3d.pybind.utility.Vector3iVector(tri)
    ''''''
    
    return mesh

def get_int_colored_mesh(mesh, label_arr):
    palet = np.array([
        [0, 0, 0],
        
        [10, 10, 10],
        [20, 20, 20],
        [30, 30, 30],
        [40, 40, 40],
        [50, 50, 50],
        [60, 60, 60],
        [70, 70, 70],
        [80, 80, 80],
        
        [90, 90, 90],
        [100, 100, 100],
        [110, 110, 110],
        [120, 120, 120],
        [130, 130, 130],
        [140, 140, 140],
        [150, 150, 150],
        [160, 160, 160]
    ])
    label_arr = label_arr.copy()
    label_arr %= palet.shape[0]
    label_colors = np.zeros((label_arr.shape[0], 3))
    for idx, palte_color in enumerate(palet):
        label_colors[label_arr==idx] = palet[idx]
    
    mesh.vertex_colors = o3d.utility.Vector3dVector(label_colors.astype(np.int64))
    return mesh

def np_to_pcd_with_label(arr, label_arr=None, axis=3):
    if type(label_arr) == np.ndarray:
        arr = np.concatenate([arr[:,:3], label_arr.reshape(-1,1)],axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:,:3])
    
    palte = np.array([
        [255,153,153],

        [153,76,0],
        [153,153,0],
        [76,153,0],
        [0,153,153],
        [0,0,153],
        [153,0,153],
        [153,0,76],
        [64,64,64],

        [255,128,0],
        [153,153,0],
        [76,153,0],
        [0,153,153],
        [0,0,153],
        [153,0,153],
        [153,0,76],
        [64,64,64],
    ])/255
    palte[9:] *= 0.4
    arr = arr.copy()
    arr[:,axis] %= palte.shape[0]
    label_colors = np.zeros((arr.shape[0], 3))
    for idx, palte_color in enumerate(palte):
        label_colors[arr[:,axis]==idx] = palte[idx]
    pcd.colors = o3d.utility.Vector3dVector(label_colors)
    return pcd

def sigmoid(x):
    return 1 / (1 +np.exp(-x))

def get_number_from_name(path):
    return int(os.path.basename(path).split("_")[-1].split(".")[0])

def get_up_from_name(path):
    return os.path.basename(path).split("_")[-1].split(".")[0]=="up"

def resample_pcd(pcd_ls, n, method):
    """Drop or duplicate points so that pcd has exactly n points"""
    if method=="uniformly":
        idx = np.random.permutation(pcd_ls[0].shape[0])
    elif method == "fps":
        idx = fps(pcd_ls[0][:,:3], n)
    pcd_resampled_ls = []
    for i in range(len(pcd_ls)):
        pcd_resampled_ls.append(pcd_ls[i][idx[:n]])
    return pcd_resampled_ls

def fps(xyz, npoint):
    if xyz.shape[0]<=npoint:
        raise "new fps error"
    xyz = torch.from_numpy(np.array(xyz)).type(torch.float).cuda()
    idx = pointops.furthestsampling(xyz, torch.tensor([xyz.shape[0]]).cuda().type(torch.int), torch.tensor([npoint]).cuda().type(torch.int)) 
    return torch_to_numpy(idx).reshape(-1)

def print_3d(*data_3d_ls):
    data_3d_ls = [item for item in data_3d_ls]
    for idx, item in enumerate(data_3d_ls):
        if type(item) == np.ndarray:
            data_3d_ls[idx] = np_to_pcd(item)
    o3d.visualization.draw_geometries(data_3d_ls, mesh_show_wireframe = True, mesh_show_back_face = True)

def torch_to_numpy(cuda_arr):
    return cuda_arr.cpu().detach().numpy()

def save_np(arr, path):
    with open(path, 'wb') as f:
        np.save(f, arr)

def load_np(path):
    with open(path, 'rb') as f:
        arr = np.load(f)
    return arr

def axis_rotation(axis, angle):
    ang = np.radians(angle) 
    R=np.zeros((3,3))
    ux, uy, uz = axis
    cos = np.cos
    sin = np.sin
    R[0][0] = cos(ang)+ux*ux*(1-cos(ang))
    R[0][1] = ux*uy*(1-cos(ang)) - uz*sin(ang)
    R[0][2] = ux*uz*(1-cos(ang)) + uy*sin(ang)
    R[1][0] = uy*ux*(1-cos(ang)) + uz*sin(ang)
    R[1][1] = cos(ang) + uy*uy*(1-cos(ang))
    R[1][2] = uy*uz*(1-cos(ang))-ux*sin(ang)
    R[2][0] = uz*ux*(1-cos(ang))-uy*sin(ang)
    R[2][1] = uz*uy*(1-cos(ang))+ux*sin(ang)
    R[2][2] = cos(ang) + uz*uz*(1-cos(ang))
    return R

def make_coord_frame(size=1):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])

def load_json(file_path):
    with open(file_path, "r") as st_json:
        return json.load(st_json)

def save_json(file_path, json_obj):
    with open(file_path, "w") as json_file:

        json.dump(json_obj, json_file)

def read_txt(file_path):
    f = open(file_path, 'r')
    path_ls = []
    while True:
        line = f.readline().split()
        if not line: break
        path_ls.append(os.path.join(os.path.dirname(file_path), line.split("\n")[0] + ".npy"))
    f.close()

    return path_ls

def read_txt_obj_ls(path, ret_mesh=False, use_tri_mesh=False):
    # In some cases, trimesh can change vertex order
    
    if use_tri_mesh:        ## trimesh 라이브러리를 통한 로딩
        
        tri_mesh_loaded_mesh = trimesh.load_mesh(path, process=False)
        vertex_ls = np.array(tri_mesh_loaded_mesh.vertices)
        tri_ls = np.array(tri_mesh_loaded_mesh.faces)+1

        '''
        ex)
        vertex_ls : (148299, 3)
        tri_ls : (296473, 3)

        '''
        
    else:                   ## 파일을 직접 line by line으로 읽어서 로딩
        f = open(path, 'r')
        vertex_ls = []
        tri_ls = []
        #vertex_color_ls = []
        while True:
            line = f.readline().split()
            if not line: break
            if line[0]=='v':
                vertex_ls.append(list(map(float,line[1:4])))
                #vertex_color_ls.append(list(map(float,line[4:7])))
            elif line[0]=='f':
                tri_verts_idxes = list(map(str,line[1:4]))
                if "//" in tri_verts_idxes[0]:
                    for i in range(len(tri_verts_idxes)):
                        tri_verts_idxes[i] = tri_verts_idxes[i].split("//")[0]
                tri_verts_idxes = list(map(int, tri_verts_idxes))
                tri_ls.append(tri_verts_idxes)
            else:
                continue
        f.close()
    
    
    ### (trimesh/line by line)으로 읽은 데이터  -->>  open3d 라이브러리의 mesh 변환
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertex_ls)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(tri_ls)-1)
    mesh.compute_vertex_normals()
    norms = np.array(mesh.vertex_normals)
    vertex_ls = np.array(vertex_ls)
    ###
    
    
    output = [np.concatenate([vertex_ls,norms], axis=1)]
    if ret_mesh:
        output.append(mesh)
    return output