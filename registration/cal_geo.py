import numpy as np
import potpourri3d as pp3d
import torch
import scipy.io as sio

def cal_geo(V,F):
    dist = torch.tensor([])
    for i in range(V.shape[0]):
        dist = torch.cat([dist,torch.tensor(pp3d.compute_distance(V,F,i)).unsqueeze(1)],dim=-1)
    # [][to point]
    return dist

# read .off file
V, F = pp3d.read_mesh('Falling202.off')  # off obj
dist = cal_geo(V,F)

numpy_dist = dist.numpy()

# store as dict
data_dict = {'dist': numpy_dist}

# save .mat file
sio.savemat("M-Falling202.mat", data_dict)

