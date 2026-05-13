import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from einops import rearrange, repeat


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

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
    def __init__(self, npts_ds):
        super(GlobalDownSample, self).__init__()
        self.npts_ds = npts_ds
        self.q_conv = nn.Conv1d(128, 128, 1, bias=False)
        self.k_conv = nn.Conv1d(128, 128, 1, bias=False)
        self.v_conv = nn.Conv1d(128, 128, 1, bias=False)
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
        neighbors = ops.group(x, self.K, self.group_type)  # (B, C, N) -> (B, C, N, K)
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
    def __init__(self):
        super(UpSample, self).__init__()
        self.q_conv = nn.Conv1d(128, 128, 1, bias=False)
        self.k_conv = nn.Conv1d(128, 128, 1, bias=False)
        self.v_conv = nn.Conv1d(128, 128, 1, bias=False)
        self.skip_link = nn.Conv1d(128, 128, 1, bias=False)
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
    def __init__(self):
        super(Embedding, self).__init__()
        self.K = 32
        self.group_type = 'center_diff'
        self.conv1 = nn.Sequential(nn.Conv2d(6, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))

    def forward(self, x):
        x_list = []
        x = group(x, self.K, self.group_type)  # (B, C=3, N) -> (B, C=6, N, K)
        x = self.conv1(x)  # (B, C=6, N, K) -> (B, C=128, N, K)
        x = self.conv2(x)  # (B, C=128, N, K) -> (B, C=64, N, K)
        x = x.max(dim=-1, keepdim=False)[0]  # (B, C=64, N, K) -> (B, C=64, N)
        x_list.append(x)
        x = group(x, self.K, self.group_type)  # (B, C=64, N) -> (B, C=128, N, K)
        x = self.conv3(x)  # (B, C=128, N, K) -> (B, C=128, N, K)
        x = self.conv4(x)  # (B, C=128, N, K) -> (B, C=64, N, K)
        x = x.max(dim=-1, keepdim=False)[0]  # (B, C=64, N, K) -> (B, C=64, N)
        x_list.append(x)
        x = torch.cat(x_list, dim=1)  # (B, C=128, N)
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

def select_neighbors(pcd, K, neighbor_type):
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
        output = neighbors  # output.shape == (B, C, N, K)
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
    
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device= device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)

def get_graph_feature_vnn(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device= device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)
class DecoderSimpleDGCNN(nn.Module):

    def __init__(self, device, k=128, emb_dims=1024):
        super(DecoderSimpleDGCNN, self).__init__()
        self.k = 50
        self.emb_dims = 512
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(self.emb_dims)
        self.bn7 = nn.BatchNorm1d(128)
        self.bn8 = nn.BatchNorm1d(64)
        self.device = device
        self.out = k
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(192+self.emb_dims, 128, kernel_size=1, bias=False))
        # self.conv7 = nn.Sequential(nn.Conv1d(192+self.emb_dims, 128, kernel_size=1, bias=False),
        #                            self.bn7,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv8 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),
        #                            self.bn8,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.dp1 = nn.Dropout(p=0.5)
        # self.conv9 = nn.Conv1d(64, self.out, kernel_size=1, bias=False)
        

    def forward(self, x, idx=None):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, idx=idx)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k, idx=idx)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k, idx=idx)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        # x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        # x = self.dp1(x)
        # x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        x = x.transpose(2, 1).contiguous()
        # x = F.log_softmax(x.view(-1,self.k), dim=-1)
        # x = x.view(batch_size, num_points, self.out)
        return x


class DecoderSimpleDGCNN_VNN(nn.Module):

    def __init__(self, device, k=128, emb_dims=1024):
        super(DecoderSimpleDGCNN, self).__init__()
        self.k = 50
        self.emb_dims = 512
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(self.emb_dims)
        self.bn7 = nn.BatchNorm1d(128)
        self.bn8 = nn.BatchNorm1d(64)
        self.device = device
        self.out = k
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(192+self.emb_dims, 128, kernel_size=1, bias=False))
        # self.conv7 = nn.Sequential(nn.Conv1d(192+self.emb_dims, 128, kernel_size=1, bias=False),
        #                            self.bn7,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv8 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),
        #                            self.bn8,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.dp1 = nn.Dropout(p=0.5)
        # self.conv9 = nn.Conv1d(64, self.out, kernel_size=1, bias=False)
        

    def forward(self, x, idx=None):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, idx=idx)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k, idx=idx)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k, idx=idx)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        # x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        # x = self.dp1(x)
        # x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        x = x.transpose(2, 1).contiguous()
        # x = F.log_softmax(x.view(-1,self.k), dim=-1)
        # x = x.view(batch_size, num_points, self.out)
        return x
class DecoderSimpleDGCNN_sample_NIE(nn.Module):
    def __init__(self, device,k =128,emb_dims=1024):
        super(DecoderSimpleDGCNN_sample_NIE, self).__init__()
        self.k = 50
        self.emb_dims = 512
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(self.emb_dims)
        self.bn7 = nn.BatchNorm1d(128)
        self.bn8 = nn.BatchNorm1d(64)
        self.device = device
        self.out = k
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(192+self.emb_dims, 128, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv8 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),
                                        self.bn8,
                                        nn.LeakyReLU(negative_slope=0.2))
                # self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Conv1d(64, self.out, kernel_size=1, bias=False)
                

    def forward(self, x,x_grid,FPS):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature_sample(x, x_grid, k=self.k,device=self.device )
# (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 9*2, num_points, k)-> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x1_grid = torch.tensor([],device=self.device)
        for i in range(batch_size):
            x1_grid = torch.cat([x1_grid,x1[i,:,FPS[i]].unsqueeze(0)],dim=0)
            
        x = get_graph_feature_sample(x1, x1_grid,k=self.k,device=self.device)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x2_grid = torch.tensor([],device=self.device)
        for i in range(batch_size):
            x2_grid = torch.cat([x2_grid,x2[i,:,FPS[i]].unsqueeze(0)],dim=0)
        x = get_graph_feature_sample(x2,x2_grid, k=self.k,device=self.device)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x3_grid = torch.tensor([],device=self.device)
        for i in range(batch_size):
            x3_grid = torch.cat([x3_grid,x3[i,:,FPS[i]].unsqueeze(0)],dim=0)
        x_grid = torch.cat((x1_grid, x2_grid, x3_grid), dim=1)      
        x_grid = self.conv6(x_grid)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x_grid = x_grid.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)    
        # x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x_grid = x_grid.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x_grid, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)
        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) > (batch_size, 256, num_points)
        # x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        x = x.transpose(2,1).contiguous()
        # x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batch_size, num_points, self.out)
        return x
class DecoderSimpleDGCNN_sample(nn.Module):
    def __init__(self, device,k =128,emb_dims=1024):
        super(DecoderSimpleDGCNN_sample, self).__init__()
        self.k = 50
        self.emb_dims = 512
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(self.emb_dims)
        self.bn7 = nn.BatchNorm1d(128)
        self.bn8 = nn.BatchNorm1d(64)
        self.device = device
        self.out = k
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(192+self.emb_dims, 128, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv7 = nn.Sequential(nn.Conv1d(192+self.emb_dims, 128, kernel_size=1, bias=False))
        #                            #self.bn7,
                                   #nn.LeakyReLU(negative_slope=0.2))
        # self.conv7 = nn.Sequential(nn.Conv1d(192+self.emb_dims, 128, kernel_size=1, bias=False))
        self.conv8 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),
                                        self.bn8,
                                        nn.LeakyReLU(negative_slope=0.2))
                # self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Conv1d(64, self.out, kernel_size=1, bias=False)
                

    def forward(self, x,x_grid,FPS):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature_sample(x, x_grid, k=self.k,device=self.device )
# (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k)-> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x1_grid = torch.tensor([],device=self.device)
        for i in range(batch_size):
            x1_grid = torch.cat([x1_grid,x1[i,:,FPS[i]].unsqueeze(0)],dim=0)
        x = get_graph_feature_sample(x1, x1_grid,k=self.k,device=self.device)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x2_grid = torch.tensor([],device=self.device)
        for i in range(batch_size):
            x2_grid = torch.cat([x2_grid,x2[i,:,FPS[i]].unsqueeze(0)],dim=0)
        x = get_graph_feature_sample(x2,x2_grid, k=self.k,device=self.device)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x3_grid = torch.tensor([],device=self.device)
        for i in range(batch_size):
            x3_grid = torch.cat([x3_grid,x3[i,:,FPS[i]].unsqueeze(0)],dim=0)
        x_grid = torch.cat((x1_grid, x2_grid, x3_grid), dim=1)      
        x_grid = self.conv6(x_grid)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x_grid = x_grid.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)    
        # x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x_grid = x_grid.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x_grid, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)
        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        # x = self.conv8(x)                       # (batch_size, 512, num_points) > (batch_size, 256, num_points)
        # x = self.dp1(x)
        # x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        x = x.transpose(2,1).contiguous()
        # x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batch_size, num_points, self.out)
        return x

# transformer_1214
# class Dgcnnattention(nn.Module):
#     def __init__(self, device,k =128,emb_dims=1024):
#         super(Dgcnnattention, self).__init__()
#         self.embedding = Embedding()
#         self.k = 50
#         self.emb_dims = 512
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.bn4 = nn.BatchNorm2d(64)
#         self.bn5 = nn.BatchNorm2d(64)
#         self.bn6 = nn.BatchNorm1d(self.emb_dims)
#         self.bn7 = nn.BatchNorm1d(128)
#         self.bn8 = nn.BatchNorm1d(64)
#         self.device = device
#         self.out = k
#         self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
#                                    self.bn1,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
#                                    self.bn2,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
#                                    self.bn3,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
#                                    self.bn4,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
#                                    self.bn5,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv6 = nn.Sequential(nn.Conv1d(192, self.emb_dims, kernel_size=1, bias=False),
#                                    self.bn6,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv7 = nn.Sequential(nn.Conv1d(192+self.emb_dims, 128, kernel_size=1, bias=False),
#                                    self.bn7,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         # self.conv7 = nn.Sequential(nn.Conv1d(192+self.emb_dims, 128, kernel_size=1, bias=False))
#         #                            #self.bn7,
#                                    #nn.LeakyReLU(negative_slope=0.2))
#         # self.conv7 = nn.Sequential(nn.Conv1d(192+self.emb_dims, 128, kernel_size=1, bias=False))
#         self.conv8 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),
#                                         self.bn8,
#                                         nn.LeakyReLU(negative_slope=0.2))
#                 # self.dp1 = nn.Dropout(p=0.5)
#         self.conv9 = nn.Conv1d(64, self.out, kernel_size=1, bias=False)

#         self.n2p_attention1 = N2PAttention(32)
#         self.n2p_attention2 = N2PAttention(32)
#         self.n2p_attention3 = N2PAttention(32)
#         self.n2p_attention4 = N2PAttention(32)

#         self.sa1 = SA_Layer(64)
#         self.sa2 = SA_Layer(64)
#         self.sa3 = SA_Layer(64)
                

#     def forward(self, x,x_grid,FPS):
#         batch_size = x.size(0)
#         num_points = x.size(2)

#         x = get_graph_feature_sample(x, x_grid, k=self.k,device=self.device )
# # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
#         x = self.conv1(x)                       # (batch_size, 3*2, num_points, k)-> (batch_size, 64, num_points, k)
#         x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
#         x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
#         x1 = self.n2p_attention1(x1)
#         x1_g = self.sa1(x1)
            
#         x = get_graph_feature_sample(x1, x1_g,k=self.k,device=self.device)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
#         x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
#         x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
#         x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
#         x2 = self.n2p_attention2(x2)
#         x2_g = self.sa2(x2)
        
#         x = get_graph_feature_sample(x2,x2_g, k=self.k,device=self.device)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
#         x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64
#         x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
#         x3 = self.n2p_attention3(x3)
#         x3_g = self.sa3(x3)

#         x_grid = torch.cat((x1_g, x2_g, x3_g), dim=1)      
#         x_grid = self.conv6(x_grid)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
#         x_grid = x_grid.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)    
#         x_grid = x_grid.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
#         x = torch.cat((x_grid, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)
#         x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)

#         x = x.transpose(2,1).contiguous()
#         # x = F.log_softmax(x.view(-1,self.k), dim=-1)
#         x = x.view(batch_size, num_points, self.out)
#         return x

class Dgcnnattention(nn.Module):
    def __init__(self, device,k =128,emb_dims=1024):
        super(Dgcnnattention, self).__init__()
        self.embedding = Embedding()
        self.k = 50
        self.emb_dims = 512
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm1d(self.emb_dims)
        self.bn4 = nn.BatchNorm1d(self.emb_dims)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn6 = nn.BatchNorm1d(128)
        self.bn7 = nn.BatchNorm1d(128)
        self.device = device
        self.out = k
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
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

        self.n2p_attention1 = N2PAttention(32)
        self.n2p_attention2 = N2PAttention(32)
        self.n2p_attention3 = N2PAttention(32)
        self.n2p_attention4 = N2PAttention(32)

        self.sa1 = SA_Layer(64)
        self.sa2 = SA_Layer(64)
        self.sa3 = SA_Layer(64)
        self.sa4 = SA_Layer(64)
                

    def forward(self, x,x_grid,FPS):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature_sample(x, x_grid, k=self.k,device=self.device ) # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k)-> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        tmp = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        
        # x1:feature
        # (batch_size, 64, num_points) -> (batch_size, 64, num_points)
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
        return x

class DecoderSimpleDGCNN_sample_backup(nn.Module):
    def __init__(self, device,k =128,emb_dims=1024):
        super(DecoderSimpleDGCNN_sample, self).__init__()
        self.k = 50
        self.emb_dims = 512
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(self.emb_dims)
        self.bn7 = nn.BatchNorm1d(128)
        self.bn8 = nn.BatchNorm1d(64)
        self.device = device
        self.out = k
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(192+self.emb_dims, 128, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv8 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),
                                        self.bn8,
                                        nn.LeakyReLU(negative_slope=0.2))
                # self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Conv1d(64, self.out, kernel_size=1, bias=False)
                

    def forward(self, x,x_grid,FPS):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature_sample(x, x_grid, k=self.k,device=self.device )
# (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 9*2, num_points, k)-> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x1_grid = torch.tensor([],device=self.device)
        for i in range(batch_size):
            x1_grid = torch.cat([x1_grid,x1[i,:,FPS[i]].unsqueeze(0)],dim=0)
            
        x = get_graph_feature_sample(x1, x1_grid,k=self.k,device=self.device)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x2_grid = torch.tensor([],device=self.device)
        for i in range(batch_size):
            x2_grid = torch.cat([x2_grid,x2[i,:,FPS[i]].unsqueeze(0)],dim=0)
        x = get_graph_feature_sample(x2,x2_grid, k=self.k,device=self.device)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x3_grid = torch.tensor([],device=self.device)
        for i in range(batch_size):
            x3_grid = torch.cat([x3_grid,x3[i,:,FPS[i]].unsqueeze(0)],dim=0)
        x = torch.cat((x1_grid, x2_grid, x3_grid), dim=1)      
        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)    
        # x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)
        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        # x = self.conv8(x)                       # (batch_size, 512, num_points) > (batch_size, 256, num_points)
        # x = self.dp1(x)
        # x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        x = x.transpose(2,1).contiguous()
        # x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batch_size, num_points, self.out)
        return x

def get_graph_feature_sample(x,x_grid, k=20, idx=None, dim9=False,device='cuda'):
    batch_size = x.size(0)
    num_points = x.size(2)
    num_points_sample = x_grid.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knnsearch(x_grid.transpose(1,2),x.transpose(1,2), k=1)   # (batch_size, num_points, k)
            idx_grid = knn(x_grid, k=k)
            idx_final = torch.tensor([],device=x.device).long()
            for i in range(batch_size):
                idx_grid_batch = idx_grid[i]
                idx_batch = idx[i].squeeze()
                idx_final = torch.cat((idx_final, idx_grid_batch[idx_batch,:].unsqueeze(0)), dim=0)
            # del idx_batch
            # del idx_grid_batch
            # del idx_grid
        else:
            idx = knn(x[:, 6:], k=k)
    idx = idx_final
    device = device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points_sample

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_siz
    x_grid = x_grid.transpose(2, 1).contiguous()
    feature = x_grid.view(batch_size*num_points_sample, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    for i in range(batch_size):
        feature[i,:,0,:] = x[i]
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature      # (batch_size, 2*num_dims, num_points, k)

def knnsearch(tar, src, k):
    bat_size = tar.shape[0]
    idx = torch.tensor([],device=tar.device).long()
    for i in range(bat_size):
        P1 = src[i]
        P2 = tar[i]
        N1,C = P1.shape
        N2,C = P2.shape
        pairwise_distance = -  torch.cdist(P1,P2)
        # print(P1.shape,P2.shape,pairwise_distance.shape)
        # exit()
        # pairwise_distance1 = -  torch.sqrt(torch.sum(torch.square(P1.view(N1,1,C) - P2.view(1,N2,C)),dim=-1))
        # print(pairwise_distance1-pairwise_distance)
        # exit()
        _ , t_st= pairwise_distance.topk(k=k, dim=-1)# (batch_size, num_points, k)

        idx = torch.cat((idx, t_st.unsqueeze(0)), dim=0)
    return idx
