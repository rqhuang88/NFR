import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import random

device='cuda:0'

def farthest_point_sample(xyz, npoint):
    xyz = xyz.unsqueeze(0)
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N,dtype=torch.float32).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def rotate_point_cloud_torch(cloud, angle, axis='z'):

    if axis == 'z':
        rotation_matrix = torch.tensor([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,               0 ,             1]
        ],device=device)
    elif axis == 'y':
        rotation_matrix = torch.tensor([
            [ np.cos(angle), 0, np.sin(angle)],
            [0              , 1,             0],
            [-np.sin(angle), 0, np.cos(angle)]
        ],device=device)
    elif axis == 'x':
        rotation_matrix = torch.tensor([
            [1,              0,             0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle),  np.cos(angle)]
        ],device=device)
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")    
    rotation_matrix = rotation_matrix.float()
    rotated_cloud = torch.matmul(cloud, rotation_matrix)    
    return rotated_cloud

def rotate_point_cloud_batch_torch(cloud, angle, axis='z'):
    # Ensure cloud is on the correct device
    # cloud = cloud.to(device)
    if axis == 'z':
        rotation_matrix = torch.tensor([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ], device=device)
    elif axis == 'y':
        rotation_matrix = torch.tensor([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ], device=device)
    elif axis == 'x':
        rotation_matrix = torch.tensor([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ], device=device)
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")    
    rotation_matrix = rotation_matrix.float()
    b, p, _ = cloud.shape
    rotation_matrix = rotation_matrix[None, :].repeat(b, 1, 1)
    # print(cloud.shape)
    # print(rotation_matrix.shape)
    rotated_cloud = torch.bmm(cloud.permute(0,2,1), rotation_matrix)    
    return rotated_cloud


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.bn1 = nn.BatchNorm1d(64)
        self.conv1 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    # def forward(self, x , phi):
    #     x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
    #     x_k = self.k_conv(x)# b, c, n        
    #     x_v = self.v_conv(x)
    #     energy = torch.bmm(x_q, x_k) # b, n, n 
    #     attention = self.softmax(energy)
    #     attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
    #     x_r = torch.bmm(x_v, attention) # b, c, n 
    #     x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
    #     phi = self.conv1(phi)
    #     x = x + x_r + phi
    #     return x

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k) # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention) # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

class GlobalDownSample(nn.Module):
    def __init__(self, npts_ds,dim_v):
        super(GlobalDownSample, self).__init__()
        self.npts_ds = npts_ds
        self.q_conv = nn.Conv1d(dim_v, dim_v, 1, bias=False)
        self.k_conv = nn.Conv1d(dim_v, dim_v, 1, bias=False)
        self.v_conv = nn.Conv1d(dim_v, dim_v, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.q_conv(x)  # (B, C, N) -> (B, C, N)
        k = self.k_conv(x)  # (B, C, N) -> (B, C, N)
        v = self.v_conv(x)  # (B, C, N) -> (B, C, N)
        energy = rearrange(q, 'B C N -> B N C').contiguous() @ k  # (B, N, C) @ (B, C, N) -> (B, N, N)
        scale_factor = math.sqrt(q.shape[-2])
        attention = self.softmax(energy / scale_factor)  # (B, N, N) -> (B, N, N)
        selection = torch.sum(attention, dim=-2)  # (B, N, N) -> (B, N)
        self.idx = selection.topk(self.npts_ds, dim=-1)[1]  # (B, N) -> (B, M)
        scores = torch.gather(attention, dim=1, index=repeat(self.idx, 'B M -> B M N', N=attention.shape[-1]))  # (B, N, N) -> (B, M, N)
        v = scores @ rearrange(v, 'B C N -> B N C').contiguous()  # (B, M, N) @ (B, N, C) -> (B, M, C)
        out = rearrange(v, 'B M C -> B C M').contiguous()  # (B, M, C) -> (B, C, M)
        return out


class LocalDownSample(nn.Module):
    def __init__(self, npts_ds):
        super(LocalDownSample, self).__init__()
        self.npts_ds = npts_ds  # number of downsampled points
        self.K = 32  # number of neighbors
        self.group_type = 'diff'
        self.q_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.k_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.v_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        neighbors = group(x, self.K, self.group_type)  # (B, C, N) -> (B, C, N, K)
        q = self.q_conv(rearrange(x, 'B C N -> B C N 1')).contiguous()  # (B, C, N) -> (B, C, N, 1)
        q = rearrange(q, 'B C N 1 -> B N 1 C').contiguous()  # (B, C, N, 1) -> (B, N, 1, C)
        k = self.k_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        k = rearrange(k, 'B C N K -> B N C K').contiguous()  # (B, C, N, K) -> (B, N, C, K)
        v = self.v_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        v = rearrange(v, 'B C N K -> B N K C').contiguous()  # (B, C, N, K) -> (B, N, K, C)
        energy = q @ k  # (B, N, 1, C) @ (B, N, C, K) -> (B, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # (B, N, 1, K) -> (B, N, 1, K)
        selection = rearrange(torch.std(attention, dim=-1, unbiased=False), 'B N 1 -> B N').contiguous()  # (B, N, 1, K) -> (B, N, 1) -> (B, N)
        self.idx = selection.topk(self.npts_ds, dim=-1)[1]  # (B, N) -> (B, M)
        scores = torch.gather(attention, dim=1, index=repeat(self.idx, 'B M -> B M 1 K', K=attention.shape[-1]))  # (B, N, 1, K) -> (B, M, 1, K)
        v = torch.gather(v, dim=1, index=repeat(self.idx, 'B M -> B M K C', K=v.shape[-2], C=v.shape[-1]))  # (B, N, K, C) -> (B, M, K, C)
        out = rearrange(scores@v, 'B M 1 C -> B C M').contiguous()  # (B, M, 1, K) @ (B, M, K, C) -> (B, M, 1, C) -> (B, C, M)
        return out


class UpSample(nn.Module):
    def __init__(self,dim_v):
        super(UpSample, self).__init__()
        self.q_conv = nn.Conv1d(dim_v, dim_v, 1, bias=False)
        self.k_conv = nn.Conv1d(dim_v, dim_v, 1, bias=False)
        self.v_conv = nn.Conv1d(dim_v, dim_v, 1, bias=False)
        self.skip_link = nn.Conv1d(dim_v, dim_v, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, pcd_up, pcd_down):
        q = self.q_conv(pcd_up)  # (B, C, N) -> (B, C, N)
        k = self.k_conv(pcd_down)  # (B, C, M) -> (B, C, M)
        v = self.v_conv(pcd_down)  # (B, C, M) -> (B, C, M)
        energy = rearrange(q, 'B C N -> B N C').contiguous() @ k  # (B, N, C) @ (B, C, M) -> (B, N, M)
        scale_factor = math.sqrt(q.shape[-2])
        attention = self.softmax(energy / scale_factor)  # (B, N, M) -> (B, N, M)
        x = attention @ rearrange(v, 'B C M -> B M C').contiguous()  # (B, N, M) @ (B, M, C) -> (B, N, C)
        x = rearrange(x, 'B N C -> B C N').contiguous()  # (B, N, C) -> (B, C, N)
        x = self.skip_link(pcd_up) + x  # (B, C, N) + (B, C, N) -> (B, C, N)
        return x

class Embedding(nn.Module):
    def __init__(self,k=32):
        super(Embedding, self).__init__()
        self.device = 'cuda:0'
        self.K = k
        self.group_type = 'center_diff'
        self.conv1 = nn.Sequential(nn.Conv2d(6, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(128, 384, 1, bias=False), nn.BatchNorm1d(384), nn.LeakyReLU(0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(384, 64, 1, bias=False), nn.BatchNorm1d(64), nn.LeakyReLU(0.2))

    def pos_encoding_sin_wave(self, coor):
        # ref to https://arxiv.org/pdf/2003.08934v2.pdf
        D = 64 #
        # normal the coor into [-1, 1], batch wise
        normal_coor = 2 * ((coor - coor.min()) / (coor.max() - coor.min())) - 1 

        # define sin wave freq
        freqs = torch.arange(D, dtype=torch.float).cuda() 
        freqs = np.pi * (2**freqs)       

        freqs = freqs.view(*[1]*len(normal_coor.shape), -1) # 1 x 1 x 1 x D
        normal_coor = normal_coor.unsqueeze(-1) # B x 3 x N x 1
        k = normal_coor * freqs # B x 3 x N x D
        s = torch.sin(k) # B x 3 x N x D
        c = torch.cos(k) # B x 3 x N x D
        x = torch.cat([s,c], -1) # B x 3 x N x 2D
        pos = x.transpose(-1,-2).reshape(coor.shape[0], -1, coor.shape[-1]) # B 6D N
        # zero_pad = torch.zeros(x.size(0), 2, x.size(-1)).cuda()
        # pos = torch.cat([x, zero_pad], dim = 1)
        # pos = self.pos_embed_wave(x)
        return pos
    
    def forward(self, x):
        batch_size = x.size(0)
        pos = self.pos_encoding_sin_wave(x)
        x_list = []
        x = group(x,self.K, self.group_type)  # (B, C=3, N) -> (B, C=6, N, K)
        x = self.conv1(x)  # (B, C=6, N, K) -> (B, C=128, N, K)
        x = self.conv2(x)  # (B, C=128, N, K) -> (B, C=64, N, K)
        x = x.max(dim=-1, keepdim=False)[0]  # (B, C=64, N, K) -> (B, C=64, N)
        x_list.append(x)
        x = group(x,self.K, self.group_type)  # (B, C=64, N) -> (B, C=128, N, K)
        x = self.conv3(x)  # (B, C=128, N, K) -> (B, C=128, N, K)
        x = self.conv4(x)  # (B, C=128, N, K) -> (B, C=64, N, K)
        x = x.max(dim=-1, keepdim=False)[0]  # (B, C=64, N, K) -> (B, C=64, N)
        x_list.append(x)
        x = torch.cat(x_list, dim=1)  # (B, C=128, N)
        x = self.conv5(x)
        x = x + pos
        x = self.conv6(x)
        return x

def index_points(points, idx):
    """
    :param points: points.shape == (B, N, C)
    :param idx: idx.shape == (B, N, K)
    :return:indexed_points.shape == (B, N, K, C)
    """
    raw_shape = idx.shape
    idx = idx.reshape(raw_shape[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.shape[-1]))
    return res.view(*raw_shape, -1)


def knn_new(a, b, k):
    """
    :param a: a.shape == (B, N, C)
    :param b: b.shape == (B, M, C)
    :param k: int
    """
    inner = -2 * torch.matmul(a, b.transpose(2, 1))  # inner.shape == (B, N, M)
    aa = torch.sum(a**2, dim=2, keepdim=True)  # aa.shape == (B, N, 1)
    bb = torch.sum(b**2, dim=2, keepdim=True)  # bb.shape == (B, M, 1)
    pairwise_distance = -aa - inner - bb.transpose(2, 1)  # pairwise_distance.shape == (B, N, M)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # idx.shape == (B, N, K)
    return idx

# def knn_new(x, y, k):
#     # distance = torch.cdist(x.float(), y.float())
#     distance = torch.cdist(x.float(), y.float(), compute_mode='donot_use_mm_for_euclid_dist')
#     _, idx = distance.topk(k=k, dim=-1, largest=False)
#     return idx

def select_neighbors(pcd, K, neighbor_type):
    #batch_size = pcd.size(0)
    pcd = pcd.permute(0, 2, 1)  # pcd.shape == (B, N, C)
    if neighbor_type == 'neighbor':
        idx = knn_new(pcd, pcd, K)  # idx.shape == (B, N, K)
        neighbors = index_points(pcd, idx)  # neighbors.shape == (B, N, K, C)
        neighbors = neighbors.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
    elif neighbor_type == 'diff':
        idx = knn_new(pcd, pcd, K)  # idx.shape == (B, N, K)
        neighbors = index_points(pcd, idx)  # neighbors.shape == (B, N, K, C)
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        neighbors = diff.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)            
    elif neighbor_type == 'grobal':
        b = pcd.shape[0]
        n = pcd.shape[1]
        k = 512
        arr = torch.zeros(b,n,k)
        indices = torch.arange(k).reshape(1,1,k)
        idx = indices.expand_as(arr).to('cuda:0')
        neighbors = index_points(pcd, idx)  # neighbors.shape == (B, N, K, C)
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        neighbors = diff.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
    else:
        raise ValueError(f'neighbor_type should be "neighbor" or "diff", but got {neighbor_type}')
    return neighbors

def select_neighbors_old(pcd, K, neighbor_type):
    pcd = pcd.permute(0, 2, 1)  # pcd.shape == (B, N, C)
    if neighbor_type == 'neighbor':
        idx = knn_new(pcd, pcd, K)  # idx.shape == (B, N, K)
        neighbors = index_points(pcd, idx)  # neighbors.shape == (B, N, K, C)
        neighbors = neighbors.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
    elif neighbor_type == 'diff':
        idx = knn_new(pcd, pcd, K)  # idx.shape == (B, N, K)
        neighbors = index_points(pcd, idx)  # neighbors.shape == (B, N, K, C)
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        neighbors = diff.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
    elif neighbor_type == 'grobal':
        b = pcd.shape[0]
        n = pcd.shape[1]
        k = 512
        arr = torch.zeros(b,n,k)
        indices = torch.arange(k).reshape(1,1,k)
        idx = indices.expand_as(arr).to('cuda:0')
        neighbors = index_points(pcd, idx)  # neighbors.shape == (B, N, K, C)
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        neighbors = diff.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
    else:
        raise ValueError(f'neighbor_type should be "neighbor" or "diff", but got {neighbor_type}')
    return neighbors

def group(pcd, K, group_type):
    if group_type == 'neighbor':
        neighbors = select_neighbors(pcd, K, 'neighbor')  # neighbors.shape == (B, C, N, K)
    elif group_type == 'diff':
        diff = select_neighbors(pcd, K, 'diff')  # diff.shape == (B, C, N, K)
        output = diff  # output.shape == (B, C, N, K)
    elif group_type == 'grobal':
        diff = select_neighbors(pcd, K, 'grobal')  # diff.shape == (B, C, N, K)
        output = diff  # output.shape == (B, C, N, K)
    elif group_type == 'center_neighbor':
        neighbors = select_neighbors(pcd, K, 'neighbor')   # neighbors.shape == (B, C, N, K)
        output = torch.cat([pcd[:, :, :, None].repeat(1, 1, 1, K), neighbors], dim=1)  # output.shape == (B, 2C, N, K)
    elif group_type == 'center_diff':
        diff = select_neighbors(pcd, K, 'diff')  # diff.shape == (B, C, N, K)
        output = torch.cat([pcd[:, :, :, None].repeat(1, 1, 1, K), diff], dim=1)  # output.shape == (B, 2C, N, K)
    else:
        raise ValueError(f'group_type should be neighbor, diff, center_neighbor or center_diff, but got {group_type}')
    return output.contiguous()

class N2PAttention(nn.Module):
    def __init__(self,k):
        super(N2PAttention, self).__init__()
        self.heads = 4
        self.K = k
        self.group_type = 'diff'
        self.q_conv = nn.Conv2d(64, 64, 1, bias=False)
        self.k_conv = nn.Conv2d(64, 64, 1, bias=False)
        self.v_conv = nn.Conv2d(64, 64, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Conv1d(64, 256, 1, bias=False), nn.LeakyReLU(0.2), nn.Conv1d(256, 64, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x,verts_sample, fps):
        neighbors = group(x,verts_sample,fps,self.K, self.group_type,sample=False)  # (B, C, N) -> (B, C, N, K)
        q = self.q_conv(rearrange(x, 'B C N -> B C N 1')).contiguous()  # (B, C, N) -> (B, C, N, 1)
        q = self.split_heads(q, self.heads)  # (B, C, N, 1) -> (B, H, N, 1, D)
        k = self.k_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        k = self.split_heads(k, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        v = self.v_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        v = self.split_heads(v, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        energy = q @ rearrange(k, 'B H N K D -> B H N D K').contiguous()  # (B, H, N, 1, D) @ (B, H, N, D, K) -> (B, H, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # (B, H, N, 1, K) -> (B, H, N, 1, K)
        tmp = rearrange(attention@v, 'B H N 1 D -> B (H D) N').contiguous()  # (B, H, N, 1, K) @ (B, H, N, K, D) -> (B, H, N, 1, D) -> (B, C=H*D, N)
        x = self.bn1(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        tmp = self.ff(x)  # (B, C, N) -> (B, C, N)
        x = self.bn2(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        return x

    @staticmethod
    def split_heads(x, heads):
        x = rearrange(x, 'B (H D) N K -> B H N K D', H=heads).contiguous()  # (B, C, N, K) -> (B, H, N, K, D)
        return x

class N2PAttention(nn.Module):
    def __init__(self,k):
        super(N2PAttention, self).__init__()
        self.heads = 4
        self.K = k
        self.group_type = 'diff'
        self.q_conv = nn.Conv2d(64, 64, 1, bias=False)
        self.k_conv = nn.Conv2d(64, 64, 1, bias=False)
        self.v_conv = nn.Conv2d(64, 64, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Conv1d(64, 256, 1, bias=False), nn.LeakyReLU(0.2), nn.Conv1d(256, 64, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        neighbors = group(x,self.K, self.group_type)  # (B, C, N) -> (B, C, N, K)
        q = self.q_conv(rearrange(x, 'B C N -> B C N 1')).contiguous()  # (B, C, N) -> (B, C, N, 1)
        q = self.split_heads(q, self.heads)  # (B, C, N, 1) -> (B, H, N, 1, D)
        k = self.k_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        k = self.split_heads(k, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        v = self.v_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        v = self.split_heads(v, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        energy = q @ rearrange(k, 'B H N K D -> B H N D K').contiguous()  # (B, H, N, 1, D) @ (B, H, N, D, K) -> (B, H, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # (B, H, N, 1, K) -> (B, H, N, 1, K)
        tmp = rearrange(attention@v, 'B H N 1 D -> B (H D) N').contiguous()  # (B, H, N, 1, K) @ (B, H, N, K, D) -> (B, H, N, 1, D) -> (B, C=H*D, N)
        x = self.bn1(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        tmp = self.ff(x)  # (B, C, N) -> (B, C, N)
        x = self.bn2(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        return x

    @staticmethod
    def split_heads(x, heads):
        x = rearrange(x, 'B (H D) N K -> B H N K D', H=heads).contiguous()  # (B, C, N, K) -> (B, H, N, K, D)
        return x

class N2PAttention_DIM(nn.Module):
    def __init__(self,k):
        super(N2PAttention_DIM, self).__init__()
        self.heads = 4
        self.K = k
        self.group_type = 'diff'
        self.q_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.k_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.v_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Conv1d(128, 512, 1, bias=False), nn.LeakyReLU(0.2), nn.Conv1d(512, 128, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        neighbors = group(x,self.K, self.group_type)  # (B, C, N) -> (B, C, N, K)
        q = self.q_conv(rearrange(x, 'B C N -> B C N 1')).contiguous()  # (B, C, N) -> (B, C, N, 1)
        q = self.split_heads(q, self.heads)  # (B, C, N, 1) -> (B, H, N, 1, D)
        k = self.k_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        k = self.split_heads(k, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        v = self.v_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        v = self.split_heads(v, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        energy = q @ rearrange(k, 'B H N K D -> B H N D K').contiguous()  # (B, H, N, 1, D) @ (B, H, N, D, K) -> (B, H, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # (B, H, N, 1, K) -> (B, H, N, 1, K)
        tmp = rearrange(attention@v, 'B H N 1 D -> B (H D) N').contiguous()  # (B, H, N, 1, K) @ (B, H, N, K, D) -> (B, H, N, 1, D) -> (B, C=H*D, N)
        x = self.bn1(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        tmp = self.ff(x)  # (B, C, N) -> (B, C, N)
        x = self.bn2(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        return x

    @staticmethod
    def split_heads(x, heads):
        x = rearrange(x, 'B (H D) N K -> B H N K D', H=heads).contiguous()  # (B, C, N, K) -> (B, H, N, K, D)
        return x
        
class P2PAttention(nn.Module):
    def __init__(self):
        super(P2PAttention, self).__init__()
        self.heads = 4
        self.K = 512
        self.group_type = 'grobal'
        self.q_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.k_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.v_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Conv1d(128, 512, 1, bias=False), nn.LeakyReLU(0.2), nn.Conv1d(512, 128, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        neighbors = group(x, self.K, self.group_type)  # (B, C, N) -> (B, C, N, K)
        q = self.q_conv(rearrange(x, 'B C N -> B C N 1')).contiguous()  # (B, C, N) -> (B, C, N, 1)
        q = self.split_heads(q, self.heads)  # (B, C, N, 1) -> (B, H, N, 1, D)
        k = self.k_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        k = self.split_heads(k, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        v = self.v_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        v = self.split_heads(v, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        energy = q @ rearrange(k, 'B H N K D -> B H N D K').contiguous()  # (B, H, N, 1, D) @ (B, H, N, D, K) -> (B, H, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # (B, H, N, 1, K) -> (B, H, N, 1, K)
        tmp = rearrange(attention@v, 'B H N 1 D -> B (H D) N').contiguous()  # (B, H, N, 1, K) @ (B, H, N, K, D) -> (B, H, N, 1, D) -> (B, C=H*D, N)
        x = self.bn1(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        tmp = self.ff(x)  # (B, C, N) -> (B, C, N)
        x = self.bn2(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        return x
    
    @staticmethod
    def split_heads(x, heads):
        x = rearrange(x, 'B (H D) N K -> B H N K D', H=heads).contiguous()  # (B, C, N, K) -> (B, H, N, K, D)
        return x

# class LGAttention(nn.Module):
#     def __init__(self, k=32,emb_dims=512):
#         super(LGAttention, self).__init__()
#         self.k = k
#         self.device = 'cuda:0'
#         self.emb_dims = emb_dims
#         # self.bn1 = nn.BatchNorm1d(64)
#         # self.bn2 = nn.BatchNorm2d(64)
#         self.embedding = Embedding(self.k)
#         self.bn3 = nn.BatchNorm1d(self.emb_dims)
#         self.bn4 = nn.BatchNorm1d(self.emb_dims)
#         self.bn5 = nn.BatchNorm1d(128)
#         self.bn6 = nn.BatchNorm1d(128)
#         self.bn7 = nn.BatchNorm1d(128)
#         self.out = 128
#         # self.conv1 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),
#         #                            self.bn1,
#         #                            nn.LeakyReLU(negative_slope=0.2))
#         # self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
#         #                            self.bn2,
#         #                            nn.LeakyReLU(negative_slope=0.2))
#         self.conv3 = nn.Sequential(nn.Conv1d(256, self.emb_dims, kernel_size=1, bias=False),
#                                    self.bn3,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv4 = nn.Sequential(nn.Conv1d(256, self.emb_dims, kernel_size=1, bias=False),
#                                    self.bn4,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv5 = nn.Sequential(nn.Conv1d(256+self.emb_dims, 128, kernel_size=1, bias=False),
#                                    self.bn5,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv6 = nn.Sequential(nn.Conv1d(256+self.emb_dims, 128, kernel_size=1, bias=False),
#                                    self.bn6,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv7 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
#                                         self.bn7,
#                                         nn.LeakyReLU(negative_slope=0.2))

#         self.n2p_attention1 = N2PAttention(self.k)
#         self.n2p_attention2 = N2PAttention(self.k)
#         self.n2p_attention3 = N2PAttention(self.k)
#         self.n2p_attention4 = N2PAttention(self.k)

#         self.sa1 = SA_Layer(64)
#         self.sa2 = SA_Layer(64)
#         self.sa3 = SA_Layer(64)
#         self.sa4 = SA_Layer(64)

#     def forward(self, x,verts_sample,fps):
#         batch_size = x.size(0)
#         num_points = x.size(2)  
#         #embedding method
#         tmp = self.embedding(x,verts_sample,fps)  # (B, 3, num_points) -> (B, 64, num_points)
        
#         x1_grid = torch.tensor([],device=self.device)
#         for i in range(batch_size):
#             x1_grid = torch.cat([x1_grid,tmp[i,:,fps[i]].unsqueeze(0)],dim=0)

#         # x1:feature
#         x1 = self.n2p_attention1(tmp,x1_grid,fps)
#         x1_g = self.sa1(tmp)

#         x2_grid = torch.tensor([],device=self.device)
#         for i in range(batch_size):
#             x2_grid = torch.cat([x2_grid,x1[i,:,fps[i]].unsqueeze(0)],dim=0)
            
#         x2 = self.n2p_attention2(x1,x2_grid,fps)
#         x2_g = self.sa2(x1_g)

#         x3_grid = torch.tensor([],device=self.device)
#         for i in range(batch_size):
#             x3_grid = torch.cat([x3_grid,x2[i,:,fps[i]].unsqueeze(0)],dim=0)
            
#         x3 = self.n2p_attention3(x2,x3_grid,fps)
#         x3_g = self.sa3(x2_g)

#         x4_grid = torch.tensor([],device=self.device)
#         for i in range(batch_size):
#             x4_grid = torch.cat([x4_grid,x3[i,:,fps[i]].unsqueeze(0)],dim=0)

#         x4 = self.n2p_attention4(x3,x4_grid,fps)
#         x4_g = self.sa4(x3_g)

#         x = torch.cat((x1_grid, x2_grid, x3_grid, x4_grid), dim=1)
#         x_g = torch.cat((x1_g, x2_g, x3_g, x4_g), dim=1)

#         x = self.conv3(x)    # (batch_size, 64*4, num_points) -> (batch_size, emb_dims, num_points)
#         x_g = self.conv4(x_g)    # (batch_size, 64*4, num_points) -> (batch_size, emb_dims, num_points)

#         x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)    
#         x_g = x_g.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

#         x = x.repeat(1, 1, num_points)          # (batch_size, emb_dims, num_points) 
#         x_g = x_g.repeat(1, 1, num_points)          # (batch_size, emb_dims, num_points) 

#         x = torch.cat((x, x1, x2, x3,x4), dim=1)   # (batch_size, 1024+64*4, num_points)
#         x_g = torch.cat((x_g, x1_g, x2_g, x3_g,x4_g), dim=1)   # (batch_size, 1024+64*4, num_points)

#         x = self.conv5(x)                       # (batch_size, 1024+64*4, num_points) -> (batch_size, 128, num_points)
#         x_g = self.conv6(x_g)                   # (batch_size, 1024+64*4, num_points) -> (batch_size, 128, num_points)

#         x = torch.cat((x, x_g), dim=1)   # (batch_size, 128*2, num_points)
#         x = self.conv7(x)


#         x = x.transpose(2,1).contiguous()
#         x = x.view(batch_size, num_points, self.out)
#         return x               

class LGAttention(nn.Module):
    def __init__(self, kembed=32, k= 32,emb_dims=512):
        super(LGAttention, self).__init__()
        self.kembed = kembed
        self.k = k
        self.device = 'cuda:0'
        self.emb_dims = emb_dims
        self.embedding = Embedding(self.kembed)
        self.bn1 = nn.BatchNorm1d(self.emb_dims)
        self.bn2 = nn.BatchNorm1d(self.emb_dims)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(128)
        self.out = 128
        self.conv1 = nn.Sequential(nn.Conv1d(320, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(320, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(320+self.emb_dims, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(320+self.emb_dims, 128, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                        self.bn5,
                                        nn.LeakyReLU(negative_slope=0.2))

        self.n2p_attention1 = N2PAttention(self.k)
        self.n2p_attention2 = N2PAttention(self.k)
        self.n2p_attention3 = N2PAttention(self.k)
        self.n2p_attention4 = N2PAttention(self.k)
        self.n2p_attention5 = N2PAttention(self.k)

        self.sa1 = SA_Layer(64)
        self.sa2 = SA_Layer(64)
        self.sa3 = SA_Layer(64)
        self.sa4 = SA_Layer(64)
        self.sa5 = SA_Layer(64)
        
    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)  

        #embedding method
        tmp = self.embedding(x)  # (B, 3, num_points) -> (B, 64, num_points)       
        # x1:feature
        x1 = self.n2p_attention1(tmp)
        x1_g = self.sa1(tmp)

        x2 = self.n2p_attention2(x1)
        x2_g = self.sa2(x1_g)

        x3 = self.n2p_attention3(x2)
        x3_g = self.sa3(x2_g)

        x4 = self.n2p_attention4(x3)
        x4_g = self.sa4(x3_g)
        
        x5 = self.n2p_attention5(x4)
        x5_g = self.sa5(x4_g)
        
        x = torch.cat((x1, x2, x3, x4,x5), dim=1)
        x_g = torch.cat((x1_g, x2_g, x3_g, x4_g,x5_g), dim=1)

        x = self.conv1(x)    # (batch_size, 64*4, num_points) -> (batch_size, emb_dims, num_points)
        x_g = self.conv2(x_g)    # (batch_size, 64*4, num_points) -> (batch_size, emb_dims, num_points)

        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)    
        x_g = x_g.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, emb_dims, num_points) 
        x_g = x_g.repeat(1, 1, num_points)          # (batch_size, emb_dims, num_points) 

        x = torch.cat((x, x1, x2, x3,x4,x5), dim=1)   # (batch_size, 1024+64*4, num_points)
        x_g = torch.cat((x_g, x1_g, x2_g, x3_g,x4_g,x5_g), dim=1)   # (batch_size, 1024+64*4, num_points)

        x = self.conv3(x)                       # (batch_size, 1024+64*4, num_points) -> (batch_size, 128, num_points)
        x_g = self.conv4(x_g)                   # (batch_size, 1024+64*4, num_points) -> (batch_size, 128, num_points)

        x = torch.cat((x, x_g), dim=1)   # (batch_size, 128*2, num_points)
        x = self.conv5(x)


        x = x.transpose(2,1).contiguous()
        x = x.view(batch_size, num_points, self.out)
        return x


class LGAttention_cross_new(nn.Module):
    def __init__(self, kembed=40, k= 40,emb_dims=512):
        super(LGAttention_cross_new, self).__init__()
        self.kembed = kembed
        self.k = k
        self.device = 'cuda:0'
        self.emb_dims = emb_dims
        self.embedding = Embedding(self.kembed)
        self.bn1 = nn.BatchNorm1d(self.emb_dims)
        self.bn2 = nn.BatchNorm1d(self.emb_dims)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn6 = nn.BatchNorm1d(128)
        self.out = 128
        self.conv1 = nn.Sequential(nn.Conv1d(256, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(256, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(256+self.emb_dims, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(256+self.emb_dims, 128, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                        self.bn5,
                                        nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(384, 128, kernel_size=1, bias=False),
                                        self.bn6,
                                        nn.LeakyReLU(negative_slope=0.2))

        self.n2p_attention1 = N2PAttention(self.k)
        self.n2p_attention2 = N2PAttention(self.k)
        self.n2p_attention3 = N2PAttention(self.k)
        self.n2p_attention4 = N2PAttention(self.k)
        self.n2p_attention5 = N2PAttention_DIM(self.k)
        self.n2p_attention6 = N2PAttention_DIM(self.k)

        self.sa1 = SA_Layer(64)
        self.sa2 = SA_Layer(64)
        self.sa3 = SA_Layer(64)
        self.sa4 = SA_Layer(64)
        # self.sa5 = SA_Layer(64)
        
    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)  

        #embedding method
        tmp = self.embedding(x)  # (B, 3, num_points) -> (B, 64, num_points)       
        # x1:feature
        x1 = self.n2p_attention1(tmp)
        x1_g = self.sa1(tmp)

        x2 = self.n2p_attention2(x1)
        x2_g = self.sa2(x1_g)

        x3 = self.n2p_attention3(x2)
        x3_g = self.sa3(x2_g)

        x4 = self.n2p_attention4(x3)
        x4_g = self.sa4(x3_g)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x_g = torch.cat((x1_g, x2_g, x3_g, x4_g), dim=1)

        x = self.conv1(x)    # (batch_size, 64*4, num_points) -> (batch_size, emb_dims, num_points)
        x_g = self.conv2(x_g)    # (batch_size, 64*4, num_points) -> (batch_size, emb_dims, num_points)

        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)    
        x_g = x_g.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, emb_dims, num_points) 
        x_g = x_g.repeat(1, 1, num_points)          # (batch_size, emb_dims, num_points) 

        x = torch.cat((x, x1, x2, x3,x4), dim=1)   # (batch_size, 1024+64*4, num_points)
        x_g = torch.cat((x_g, x1_g, x2_g, x3_g,x4_g), dim=1)   # (batch_size, 1024+64*4, num_points)

        x = self.conv3(x)                       # (batch_size, 1024+64*4, num_points) -> (batch_size, 128, num_points)
        x_g = self.conv4(x_g)                   # (batch_size, 1024+64*4, num_points) -> (batch_size, 128, num_points)

        x = torch.cat((x, x_g), dim=1)   # (batch_size, 128*2, num_points)
        x_1 = self.conv5(x)
        x_2 = self.n2p_attention5(x_1)
        x_3 = self.n2p_attention6(x_2)

        x = torch.cat((x_1, x_2, x_3), dim=1) 
        x = self.conv6(x)

        x = x.transpose(2,1).contiguous()
        x = x.view(batch_size, num_points, self.out)
        return x
    
class LGAttention_sample_depth(nn.Module):
    def __init__(self, kembed=32, k= 32,emb_dims=512):
        super(LGAttention_sample_depth, self).__init__()
        self.kembed = kembed
        self.k = k
        self.device = 'cuda:0'
        self.emb_dims = emb_dims
        # self.bn1 = nn.BatchNorm1d(64)
        # self.bn2 = nn.BatchNorm2d(64)
        self.embedding = Embedding(self.kembed)
        self.bn3 = nn.BatchNorm1d(self.emb_dims)
        self.bn4 = nn.BatchNorm1d(self.emb_dims)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn6 = nn.BatchNorm1d(128)
        self.bn7 = nn.BatchNorm1d(128)
        # self.bn8 = nn.BatchNorm1d(128)
        self.out = 128
        # self.conv1 = nn.Sequential(nn.Conv1d(192, 64, kernel_size=1, bias=False),
        #                            self.bn1,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
        #                            self.bn2,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(384, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(384, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(384+self.emb_dims, 128, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(384+self.emb_dims, 128, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                        self.bn7,
                                        nn.LeakyReLU(negative_slope=0.2))
        # self.conv8 = nn.Sequential(nn.Conv1d(384, 128, kernel_size=1, bias=False),
        #                                 self.bn8,
        #                                 nn.LeakyReLU(negative_slope=0.2))

        self.n2p_attention1 = N2PAttention(self.k)
        self.n2p_attention2 = N2PAttention(self.k)
        self.n2p_attention3 = N2PAttention(self.k)
        self.n2p_attention4 = N2PAttention(self.k)
        self.n2p_attention5 = N2PAttention(self.k)
        self.n2p_attention6 = N2PAttention(self.k)

        self.sa1 = SA_Layer(64)
        self.sa2 = SA_Layer(64)
        self.sa3 = SA_Layer(64)
        self.sa4 = SA_Layer(64)
        self.sa5 = SA_Layer(64)
        self.sa6 = SA_Layer(64)
        
    def forward(self, x,verts_sample,fps):
        batch_size = x.size(0)
        num_points = x.size(2)  

        #embedding method
        # x1 = rotate_point_cloud_batch_torch(x, math.pi/4, axis='y').permute(0,2,1)
        # x2 = rotate_point_cloud_batch_torch(x, -math.pi/4, axis='y').permute(0,2,1)
        tmp = self.embedding(x,None,None)  # (B, 3, num_points) -> (B, 64, num_points)
        # tmp1 = self.embedding(x1,None,None)  # (B, 3, num_points) -> (B, 64, num_points)
        # tmp2 = self.embedding(x2,None,None)  # (B, 3, num_points) -> (B, 64, num_points)
        
        # tmp = torch.cat((tmp, tmp1, tmp2), dim=1)   # (batch_size, 192, num_points)
        # tmp = self.conv1(tmp)        
        
        # x1:feature
        x1 = self.n2p_attention1(tmp,None,None)
        x1_g = self.sa1(tmp)

        x2 = self.n2p_attention2(x1,None,None)
        x2_g = self.sa2(x1_g)

        x3 = self.n2p_attention3(x2,None,None)
        x3_g = self.sa3(x2_g)

        x4 = self.n2p_attention4(x3,None,None)
        x4_g = self.sa4(x3_g)
        
        x5 = self.n2p_attention5(x4,None,None)
        x5_g = self.sa5(x4_g)

        x6 = self.n2p_attention6(x5,None,None)
        x6_g = self.sa6(x5_g)

        x = torch.cat((x1, x2, x3, x4,x5,x6), dim=1)
        x_g = torch.cat((x1_g, x2_g, x3_g, x4_g,x5_g,x6_g), dim=1)

        x = self.conv3(x)    # (batch_size, 64*4, num_points) -> (batch_size, emb_dims, num_points)
        x_g = self.conv4(x_g)    # (batch_size, 64*4, num_points) -> (batch_size, emb_dims, num_points)

        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)    
        x_g = x_g.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, emb_dims, num_points) 
        x_g = x_g.repeat(1, 1, num_points)          # (batch_size, emb_dims, num_points) 

        x = torch.cat((x, x1, x2, x3,x4,x5,x6), dim=1)   # (batch_size, 1024+64*4, num_points)
        x_g = torch.cat((x_g, x1_g, x2_g, x3_g,x4_g,x5_g,x6_g), dim=1)   # (batch_size, 1024+64*4, num_points)

        x = self.conv5(x)                       # (batch_size, 1024+64*4, num_points) -> (batch_size, 128, num_points)
        x_g = self.conv6(x_g)                   # (batch_size, 1024+64*4, num_points) -> (batch_size, 128, num_points)

        x = torch.cat((x, x_g), dim=1)   # (batch_size, 128*2, num_points)
        x = self.conv7(x)

        x = x.transpose(2,1).contiguous()
        x = x.view(batch_size, num_points, self.out)
        return x,tmp.permute(0,2,1)


class cross_transformer(nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)

        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)

        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.activation1 = torch.nn.GELU()

        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src1, src2, if_act=False):
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)

        b, c, _ = src1.shape

        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)

        src1 = self.norm13(src1)
        src2 = self.norm13(src2)

        src12 = self.multihead_attn1(query=src1,
                                     key=src2,
                                     value=src2)[0]


        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)

        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)


        src1 = src1.permute(1, 2, 0)

        return src1

        # self.sa1_1 = cross_transformer(128,128)
class Embedding_new(nn.Module):
    def __init__(self,k=32):
        super(Embedding_new, self).__init__()
        self.device = 'cuda:0'
        self.K = k
        self.group_type = 'center_diff'
        self.conv1 = nn.Sequential(nn.Conv2d(6, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(128, 384, 1, bias=False), nn.BatchNorm1d(384), nn.LeakyReLU(0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(384, 64, 1, bias=False), nn.BatchNorm1d(64), nn.LeakyReLU(0.2))
        self.sa1_1 = cross_transformer(128,128)

    def pos_encoding_sin_wave(self, coor):
        # ref to https://arxiv.org/pdf/2003.08934v2.pdf
        D = 64 #
        # normal the coor into [-1, 1], batch wise
        normal_coor = 2 * ((coor - coor.min()) / (coor.max() - coor.min())) - 1 

        # define sin wave freq
        freqs = torch.arange(D, dtype=torch.float).cuda() 
        freqs = np.pi * (2**freqs)       

        freqs = freqs.view(*[1]*len(normal_coor.shape), -1) # 1 x 1 x 1 x D
        normal_coor = normal_coor.unsqueeze(-1) # B x 3 x N x 1
        k = normal_coor * freqs # B x 3 x N x D
        s = torch.sin(k) # B x 3 x N x D
        c = torch.cos(k) # B x 3 x N x D
        x = torch.cat([s,c], -1) # B x 3 x N x 2D
        pos = x.transpose(-1,-2).reshape(coor.shape[0], -1, coor.shape[-1]) # B 6D N
        # zero_pad = torch.zeros(x.size(0), 2, x.size(-1)).cuda()
        # pos = torch.cat([x, zero_pad], dim = 1)
        # pos = self.pos_embed_wave(x)
        return pos
    
    def forward(self, x,verts_sample,fps):
        batch_size = x.size(0)
        pos = self.pos_encoding_sin_wave(x)
        x_list = []
        x = group(x,verts_sample, fps,self.K, self.group_type,sample=True)  # (B, C=3, N) -> (B, C=6, N, K)
        x = self.conv1(x)  # (B, C=6, N, K) -> (B, C=128, N, K)
        x = self.conv2(x)  # (B, C=128, N, K) -> (B, C=64, N, K)
        x = x.max(dim=-1, keepdim=False)[0]  # (B, C=64, N, K) -> (B, C=64, N)
        x_list.append(x)
        verts_sample = torch.tensor([],device=self.device)
        for i in range(batch_size):
            verts_sample = torch.cat([verts_sample,x[i,:,fps[i]].unsqueeze(0)],dim=0)
        x = group(x, verts_sample,fps,self.K, self.group_type,sample=True)  # (B, C=64, N) -> (B, C=128, N, K)
        x = self.conv3(x)  # (B, C=128, N, K) -> (B, C=128, N, K)
        x = self.conv4(x)  # (B, C=128, N, K) -> (B, C=64, N, K)
        x = x.max(dim=-1, keepdim=False)[0]  # (B, C=64, N, K) -> (B, C=64, N)
        x_list.append(x)
        x = torch.cat(x_list, dim=1)  # (B, C=128, N)
        x = self.conv5(x)
        x = x + pos
        x = self.conv6(x)
        return x
    
class LGAttention_sample_new(nn.Module):
    def __init__(self, kembed=32, k= 32,emb_dims=512):
        super(LGAttention_sample_new, self).__init__()
        self.kembed = kembed
        self.k = k
        self.device = 'cuda:0'
        self.emb_dims = emb_dims
        # self.bn1 = nn.BatchNorm1d(64)
        # self.bn2 = nn.BatchNorm2d(64)
        self.embedding = Embedding_new(self.kembed)
        self.bn3 = nn.BatchNorm1d(self.emb_dims)
        self.bn4 = nn.BatchNorm1d(self.emb_dims)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn6 = nn.BatchNorm1d(128)
        self.bn7 = nn.BatchNorm1d(128)
        self.out = 128
        # self.conv1 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),
        #                            self.bn1,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
        #                            self.bn2,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(256, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(256, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(256+self.emb_dims, 128, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(256+self.emb_dims, 128, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                        self.bn7,
                                        nn.LeakyReLU(negative_slope=0.2))

        self.n2p_attention1 = N2PAttention(self.k)
        self.n2p_attention2 = N2PAttention(self.k)
        self.n2p_attention3 = N2PAttention(self.k)
        self.n2p_attention4 = N2PAttention(self.k)

        # self.sa1_1 = cross_transformer(64,64)
        # self.sa1_2 = cross_transformer(64,64)
        # self.sa1_3 = cross_transformer(64,64)
        # self.sa1_4 = cross_transformer(64,64)
        self.sa1 = SA_Layer(64)
        self.sa2 = SA_Layer(64)
        self.sa3 = SA_Layer(64)
        self.sa4 = SA_Layer(64)
        
    def forward(self, x,verts_sample,fps):
        batch_size = x.size(0)
        num_points = x.size(2)  
        
        # #embedding method
        tmp = self.embedding(x,verts_sample,fps)  # (B, 3, num_points) -> (B, 64, num_points)
        # tmp = self.embedding(x,verts_sample,fps)  # (B, 3, num_points) -> (B, 64, num_points)
        x1_grid = torch.tensor([],device=self.device)
        for i in range(batch_size):
            x1_grid = torch.cat([x1_grid,tmp[i,:,fps[i]].unsqueeze(0)],dim=0)
        # x1:feature
        x1 = self.n2p_attention1(tmp,x1_grid,fps)
        x1_g = self.sa1(tmp)

        x2_grid = torch.tensor([],device=self.device)
        for i in range(batch_size):
            x2_grid = torch.cat([x2_grid,x1[i,:,fps[i]].unsqueeze(0)],dim=0)
        x2 = self.n2p_attention2(x1,x2_grid,fps)
        x2_g = self.sa2(x1_g)

        x3_grid = torch.tensor([],device=self.device)
        for i in range(batch_size):
            x3_grid = torch.cat([x3_grid,x2[i,:,fps[i]].unsqueeze(0)],dim=0)
            
        x3 = self.n2p_attention3(x2,x3_grid,fps)
        x3_g = self.sa3(x2_g)

        x4_grid = torch.tensor([],device=self.device)
        for i in range(batch_size):
            x4_grid = torch.cat([x4_grid,x3[i,:,fps[i]].unsqueeze(0)],dim=0)

        x4 = self.n2p_attention4(x3,x4_grid,fps)
        x4_g = self.sa4(x3_g)

        x = torch.cat((x1_grid, x2_grid, x3_grid, x4_grid), dim=1)
        x_g = torch.cat((x1_g, x2_g, x3_g, x4_g), dim=1)

        x = self.conv3(x)    # (batch_size, 64*4, num_points) -> (batch_size, emb_dims, num_points)
        x_g = self.conv4(x_g)    # (batch_size, 64*4, num_points) -> (batch_size, emb_dims, num_points)

        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)    
        x_g = x_g.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, emb_dims, num_points) 
        x_g = x_g.repeat(1, 1, num_points)          # (batch_size, emb_dims, num_points) 

        x = torch.cat((x, x1, x2, x3,x4), dim=1)   # (batch_size, 1024+64*4, num_points)
        x_g = torch.cat((x_g, x1_g, x2_g, x3_g,x4_g), dim=1)   # (batch_size, 1024+64*4, num_points)

        x = self.conv5(x)                       # (batch_size, 1024+64*4, num_points) -> (batch_size, 128, num_points)
        x_g = self.conv6(x_g)                   # (batch_size, 1024+64*4, num_points) -> (batch_size, 128, num_points)

        x = torch.cat((x, x_g), dim=1)   # (batch_size, 128*2, num_points)
        x = self.conv7(x)


        x = x.transpose(2,1).contiguous()
        x = x.view(batch_size, num_points, self.out)
        return x,tmp.permute(0,2,1)