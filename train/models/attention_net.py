from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim ** 0.5
    torch.cuda.empty_cache()
    prob = torch.nn.functional.softmax(scores, dim=-1)
    torch.cuda.empty_cache()
    result = torch.einsum("bhnm,bdhm->bdhn", prob, value)
    torch.cuda.empty_cache()
    return result, prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [ll(x).view(batch_dim, self.dim, self.num_heads, -1) for ll, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class CrossAttentionRefinementNet(nn.Module):
    def __init__(self, n_in=128, num_head=4, gnn_dim=512, overlap_feat_dim=32, n_layers=2, cross_sampling_ratio=0.15, attention_type="normal"):
        super().__init__()

        self.attention_type = attention_type
        if attention_type == "normal":
            additional_dim = 0
            overlap_feat_dim = n_in
        elif attention_type == "double":
            additional_dim = overlap_feat_dim
        else:
            raise Exception("Attention type not recognized")

        self.n_in = n_in
        self.cross_sampling_ratio = cross_sampling_ratio
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(AttentionalPropagation(gnn_dim + additional_dim, num_head))
        self.layers = nn.ModuleList(self.layers)

        self.first_lin = nn.Linear(n_in, gnn_dim + additional_dim)
        self.last_lin = nn.Linear(gnn_dim + additional_dim, n_in + additional_dim)

    def forward(self, features_x, features_y):
        desc0, desc1 = self.first_lin(features_x).transpose(1, 2), self.first_lin(features_y).transpose(1, 2)

        for layer in self.layers:
            # flat_coords0, flat_coords1 = coords0.squeeze(0).t(), coords1.squeeze(0).t()
            if self.cross_sampling_ratio == 1:
                desc0 = desc0 + layer(desc0, desc1)
                torch.cuda.empty_cache()
                desc1 = desc1 + layer(desc1, desc0)
                torch.cuda.empty_cache()
            # else:
            #     n0, n1 = int(self.cross_sampling_ratio * flat_coords0.shape[1]), int(self.cross_sampling_ratio * flat_coords1.shape[1])
            #     idf0, idn0, dists0 = batch["shape1"]["sample_idx"]
            #     idf1, idn1, dists1 = batch["shape2"]["sample_idx"]
            #     idf0, idn0, dists0 = idf0[:n0], idn0, dists0
            #     idf1, idn1, dists1 = idf1[:n1], idn1, dists1
            #     sampled_desc0, sampled_desc1 = desc0[:, :, idf0], desc1[:, :, idf1]
            #     sampled_desc0 = sampled_desc0 + layer(sampled_desc0, sampled_desc1)
            #     torch.cuda.empty_cache()
            #     sampled_desc1 = sampled_desc1 + layer(sampled_desc1, sampled_desc0)
            #     torch.cuda.empty_cache()
            #     desc0 = nn_interpolate(sampled_desc0.transpose(1, 2), flat_coords0.t(), dists0, idn0, idf0).transpose(1, 2)
            #     desc1 = nn_interpolate(sampled_desc1.transpose(1, 2), flat_coords1.t(), dists1, idn1, idf1).transpose(1, 2)

        augmented_features_x = self.last_lin(desc0.transpose(1, 2))
        augmented_features_y = self.last_lin(desc1.transpose(1, 2))

        ref_feat_x, ref_feat_y = augmented_features_x[:, :, :self.n_in], augmented_features_y[:, :, :self.n_in]

        return ref_feat_x, ref_feat_y