# Reference: https://github.com/POSTECH-CVLab/point-transformer

import torch
import numpy as np
import torch.nn as nn
from lib.utils import calc_ppf_gpu, group_all, point_to_node_partition, get_node_correspondences, index_select, get_node_occlusion_score, to_o3d_pcd
import torch.nn.functional as F
from model.transformer.ppftransformer import LocalPPFTransformer, PPFTransformer
from model.transformer.geotransformer import GeometricTransformer
import open3d as o3d
from .heads import *

from cpp_wrappers.pointops.functions import pointops


class RIPointTransformerLayer(nn.Module):
    '''
    Rotation-invariant point transformer layer
    '''
    def __init__(self, in_planes, out_planes, num_heads=4, nsample=16, factor=1):
        super().__init__()
        self.nsample = nsample
        self.in_planes = in_planes
        self.output_planes = out_planes
        self.num_heas = num_heads
        self.nsample = nsample
        self.factor = factor
        self.transformer = LocalPPFTransformer(input_dim=in_planes, hidden_dim=min(out_planes, 256*factor), output_dim=out_planes, num_heads=num_heads)
        

    def forward(self, pxon, mask=None) -> torch.Tensor:
        p, x, o, n, idx, ppf_r = pxon  # (n, 3), (n, c), (b), (n, 4)
        if idx is None:
            group_idx = pointops.queryandgroup(self.nsample, p, p, p, idx, o, o, return_idx=True).long() #(n, nsample)
        else:
            group_idx = idx

        node_idx = torch.from_numpy(np.arange(p.shape[0])).to(p).long()


        p_r = p[group_idx, :]
        n_r = n[group_idx, :]

        if ppf_r is None:
            ppf_r = calc_ppf_gpu(p, n, p_r, n_r)  # (n, nsample, 4)
        x = self.transformer(x, node_idx, group_idx, ppf_r)
        return [x, group_idx, ppf_r]


class TransitionDown(nn.Module):
    '''
    Down-sampling
    '''
    def __init__(self, in_planes, out_planes, num_heads=4, stride=1, nsample=16, factor=1):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        self.transformer = LocalPPFTransformer(input_dim=in_planes, hidden_dim=min(out_planes, 256*factor), output_dim=out_planes, num_heads=num_heads)

    def forward(self, pxon):
        p, x, o, n, _, _, _ = pxon  # (n, 3), (n, c), (b), (n, 3)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o).long()  # (m)

            n_p = p[idx, :]  # (m, 3)
            n_n = n[idx, :]  # (m, 3)
        else:
            n_o = o
            n_p = p
            n_n = n
            idx = torch.from_numpy(np.arange(p.shape[0])).to(n_o).long()

        group_idx = pointops.queryandgroup(self.nsample, p, n_p, p, None, o, n_o, return_idx=True).long()  # (m, nsample, 3 + 4 + c)
        c_p, c_n = p[group_idx, :], n[group_idx, :]
        ppf = calc_ppf_gpu(n_p, n_n, c_p, c_n) # (m, nsample, 4]

        x = self.transformer(x, idx, group_idx, ppf)
        return [n_p, x, n_o, n_n, None, None, idx]


class TransitionUp(nn.Module):
    '''
    Up-sampling
    '''
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2 * in_planes, in_planes), nn.LayerNorm(in_planes),
                                         nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.LayerNorm(out_planes),
                                         nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.LayerNorm(out_planes),
                                         nn.ReLU(inplace=True))

    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                x_b = x[s_i:e_i, :]
                # print()
                # print("x_b : ", x_b.shape)
                # print("x_b.sum(0, True) / cnt : ", (x_b.sum(0, True) / cnt).shape)
                # print(self.linear2)
                # print()
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1
            p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x


class RIPointTransformerBlock(nn.Module):
    '''
    Rotation-invariant point transformer block
    '''
    expansion = 1

    def __init__(self, in_planes, planes, num_heads=4, nsample=16, factor=1):
        super(RIPointTransformerBlock, self).__init__()
        self.transformer = RIPointTransformerLayer(in_planes, planes, num_heads, nsample, factor)
        self.bn2 = nn.LayerNorm(planes)

    def forward(self, pxon, mask=None):
        #print(len(pxon))
        p, x, o, n, idx, ppf_r, down_idx = pxon  # (n, 3), (n, c), (b), (n, 4)
        identity = x

        x, idx, ppf_r = self.transformer([p, x, o, n, idx, ppf_r], mask)
        #print(idx.dtype)
        x = self.bn2(x)
        x += identity
        x = F.relu(x)

        return [p, x, o, n, idx, ppf_r, down_idx]


class RIPointTransformer(nn.Module):
    # def __init__(self, blocks=[2, 3, 3, 3], block=RIPointTransformerBlock, c=1, transformer_architecture=None, with_cross_pos_embed=None, factor=1, occ_thres=0.):
    def __init__(self, blocks=[2, 3, 4, 6, 3], block=RIPointTransformerBlock, c=1, transformer_architecture=None, with_cross_pos_embed=None, factor=1, occ_thres=0.):
        super().__init__()
        self.c = c
        self.num_heads = 4
        # self.in_planes, planes = c, [64*factor, 128*factor, 256*factor, 256*factor]
        # stride, nsample = [1, 4, 4, 4], [8, 16, 16, 16]
        self.in_planes, planes = c, [32*factor, 64*factor, 128*factor, 256*factor, 512*factor]
        stride, nsample = [1, 4, 4, 4, 4], [36, 24, 24, 24, 24]
        
        ### Encoder
        self.enc1 = self._make_enc(block, planes[0], blocks[0], self.num_heads, stride=stride[0], nsample=nsample[0], factor=factor)  # (N/1, 32)
        self.enc2 = self._make_enc(block, planes[1], blocks[1], self.num_heads, stride=stride[1], nsample=nsample[1], factor=factor)  # (N/4, 64)
        self.enc3 = self._make_enc(block, planes[2], blocks[2], self.num_heads, stride=stride[2], nsample=nsample[2], factor=factor)  # (N/16, 128)
        self.enc4 = self._make_enc(block, planes[3], blocks[3], self.num_heads, stride=stride[3], nsample=nsample[3], factor=factor)  # (N/64, 256)
        self.enc5 = self._make_enc(block, planes[4], blocks[4], self.num_heads, stride=stride[4], nsample=nsample[4], factor=factor)  # (N/256, 512)
        ###


        ### Decoder
        self.dec5 = self._make_dec(block, planes[4], 2, self.num_heads, nsample=nsample[4], factor=factor, is_head=True)  # fusion p5 and p4
        self.dec4 = self._make_dec(block, planes[3], 2, self.num_heads, nsample=nsample[3], factor=factor)  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, self.num_heads, nsample=nsample[2], factor=factor)  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, self.num_heads, nsample=nsample[1], factor=factor)  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, self.num_heads, nsample=nsample[0], factor=factor)  # fusion p2 and p1
        self.nsample = nsample
        ###
        
        
        ### for Segmentation
        
        ## Single Scale Classicifation head
        # self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], 17))
        # self.mask = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], 2))
        ##
        
        ## Multi Scale Classification head
        self.cls_head = MultiHead(planes, k=17)
        self.mask_head = MultiHead(planes, k=2)
        ##

        ## Extra head (결국은 사용 X)
        # self.sem_head = MultiHead(planes, k=9)
        # self.lr_head = MultiHead(planes, k=3)
        


    def _make_enc(self, block, planes, blocks, num_heads=4, stride=1, nsample=16, factor=1):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes, num_heads, stride, nsample, factor))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, num_heads, nsample=nsample, factor=factor))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, factor=1, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample, factor=factor))
        return nn.Sequential(*layers)

    def forward(self, pxon):

        p0, x0, o0, n0 = pxon       ## p : point cloud의 (x, y, z) 좌표들
                                    ## x : point cloud의 feature들 (처음에는 1로 초기화, 모델을 거치면서 차원이 계속 바뀜)
                                    ## o : vertex 갯수
                                    ## n : vertex normals의 (x, y, z) 좌표들
        
        
        stage_list = {'inputs': [pxon]}     ## 각 stage별 pxon들을 저장하여, 디코더에 전달 및 classify하기 위한 변수
        
        
        ############################
        # encoder
        ############################
        # source pcd
        p1, x1, o1, n1, idx1, ppf1, d_idx1 = self.enc1([p0, x0, o0, n0, None, None, None])
        p2, x2, o2, n2, idx2, ppf2, d_idx2 = self.enc2([p1, x1, o1, n1, None, None, None])
        p3, x3, o3, n3, idx3, ppf3, d_idx3 = self.enc3([p2, x2, o2, n2, None, None, None])
        p4, x4, o4, n4, idx4, ppf4, d_idx4 = self.enc4([p3, x3, o3, n3, None, None, None])
        p5, x5, o5, n5, idx5, ppf5, d_idx5 = self.enc5([p4, x4, o4, n4, None, None, None])
        
        down_list = [
            # [p0, x0, o0],  # (n, 3), (n, in_feature_dims), (b)
            {'p_out': p1, 'f_out': x1, 'offset': o1},  # (n, 3), (n, base_fdims), (b) - default base_fdims = 32
            {'p_out': p2, 'f_out': x2, 'offset': o2},  # n_1
            {'p_out': p3, 'f_out': x3, 'offset': o3},  # n_2
            {'p_out': p4, 'f_out': x4, 'offset': o4},  # n_3
            {'p_out': p5, 'f_out': x5, 'offset': o5},  # n_4 - fdims = 512
        ]
        stage_list['down'] = down_list


        ##########################
        # decoder
        ##########################

        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5, n5, idx5, ppf5, None])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4, n4, idx4, ppf4, None])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3, n3, idx3, ppf3, None])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2, n2, idx2, ppf2, None])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1, n1, idx1, ppf1, None])[1]
        
        up_list = [
            {'p_out': p1, 'f_out': x1, 'offset': o1},  # n_0 = n, fdims = 32
            {'p_out': p2, 'f_out': x2, 'offset': o2},  # n_1
            {'p_out': p3, 'f_out': x3, 'offset': o3},  # n_2
            {'p_out': p4, 'f_out': x4, 'offset': o4},  # n_3
            {'p_out': p5, 'f_out': x5, 'offset': o5},  # n_4 - fdims = 512 (extracted through dec5 = mlps)
        ]
        stage_list['up'] = up_list
        
        
        '''Classification branches'''
        
        ### Classification branch 하나만 있을 때 (Single Scale)
        # cls_results = self.cls(x1)
        # return cls_results, None, None
        ###

        ### Binary mask prediction branch 추가되었을 때 (Single Scale)
        # mask_results = self.mask(x1)
        # return cls_results, mask_results, None
        ###
    
        ### Multi scale feature aggregation
        cls_results = self.cls_head(stage_list)
        mask_results = self.mask_head(stage_list)
        # sem_results = self.sem_head(stage_list)
        # lr_results = self.lr_head(stage_list)
        
        return cls_results, mask_results, None
        # return cls_results, mask_results, sem_results
        # return cls_results, mask_results, lr_results
        ###
        
        

'''단순한 Linear layer들로만 이루어진 Axes Regression 모델'''
'''TANet의 pose regressor를 참고 (p.46~47)'''
class AxesRegressor(nn.Module):
    def __init__(self, input_dim, mlps=[512, 1024, 1024, 512, 256, 128, 64, 32, 3],
                 activators=['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'tanh'],
                 dropouts=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.]):
        super().__init__()
        assert len(mlps) == len(activators) == len(dropouts)
        mlps.insert(0, input_dim)
        tmp = nn.ModuleList()
        tmp.append(nn.Flatten())
        for i, combs in enumerate(zip(activators, dropouts)):
            tmp.append(nn.Linear(mlps[i], mlps[i+1]))
            tmp.append(nn.ReLU() if combs[0] == "relu" else nn.Tanh())
            tmp.append(nn.Dropout(combs[1]))
        nn.init.zeros_(tmp[-1-2].weight.data)
        nn.init.zeros_(tmp[-1-2].bias.data)
        self.mlps = nn.Sequential(*tmp)
        
    def forward(self, x: torch.Tensor):
        x = self.mlps(x)
        
        return x
        
        

'''(Rotation-Invariant encoder + Linear layers)로 이루어진 Axes Regression 모델'''
class RIPointTransformerAxesRegressor(nn.Module):
    def __init__(self, blocks=[2, 3, 4, 6, 3], block=RIPointTransformerBlock, c=1, factor=1):
        super().__init__()
        # Backbone
        self.c = c
        self.num_heads = 4
        # self.in_planes, planes = c, [64*factor, 128*factor, 256*factor, 256*factor]
        # stride, nsample = [1, 4, 4, 4], [8, 16, 16, 16]
        self.in_planes, planes = c, [32*factor, 64*factor, 128*factor, 256*factor, 512*factor]
        stride, nsample = [1, 4, 4, 4, 4], [36, 24, 24, 24, 24]
        
        ### Encoder
        self.enc1 = self._make_enc(block, planes[0], blocks[0], self.num_heads, stride=stride[0], nsample=nsample[0], factor=factor)  # (N/1, 32)

        self.enc2 = self._make_enc(block, planes[1], blocks[1], self.num_heads, stride=stride[1], nsample=nsample[1], factor=factor)  # (N/4, 64)
        self.enc3 = self._make_enc(block, planes[2], blocks[2], self.num_heads, stride=stride[2], nsample=nsample[2], factor=factor)  # (N/16, 128)
        self.enc4 = self._make_enc(block, planes[3], blocks[3], self.num_heads, stride=stride[3], nsample=nsample[3], factor=factor)  # (N/64, 256)
        self.enc5 = self._make_enc(block, planes[4], blocks[4], self.num_heads, stride=stride[4], nsample=nsample[4], factor=factor)  # (N/256, 512)
        ###
        
        # MLPs
        self.mlps = AxesRegressor(planes[-1])
        
    
    def forward(self, pxon):
    # def forward(self, x: torch.Tensor):
        p0, x0, o0, n0 = pxon
        
        stage_list = {'inputs': [pxon]}     ## MLPs에 multi-scale features를 전달하는 생각도 해봐서 일단 놔둠
        
        
        ### Encoder
        p1, x1, o1, n1, idx1, ppf1, d_idx1 = self.enc1([p0, x0, o0, n0, None, None, None])
        p2, x2, o2, n2, idx2, ppf2, d_idx2 = self.enc2([p1, x1, o1, n1, None, None, None])
        p3, x3, o3, n3, idx3, ppf3, d_idx3 = self.enc3([p2, x2, o2, n2, None, None, None])
        p4, x4, o4, n4, idx4, ppf4, d_idx4 = self.enc4([p3, x3, o3, n3, None, None, None])
        p5, x5, o5, n5, idx5, ppf5, d_idx5 = self.enc5([p4, x4, o4, n4, None, None, None])
        ###
        
        
        x = self.mlps(x5)
        
        return x
        
        
    def _make_enc(self, block, planes, blocks, num_heads=4, stride=1, nsample=16, factor=1):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes, num_heads, stride, nsample, factor))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, num_heads, nsample=nsample, factor=factor))
        return nn.Sequential(*layers)