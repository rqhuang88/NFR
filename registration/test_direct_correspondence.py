
import argparse
import yaml
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
import torch
import sys
import time
import threading
import random

import numpy as np
import scipy.io as sio
import torch.nn.functional as F
import open3d as o3d
from tqdm import tqdm

from dataset import testDataset, shape_to_device
from models.dgcnn import DecoderSimpleDGCNN_sample
from utils import read_geodist, calculate_geodesic_error, compute_geodesic_distmat, search_t, compute_vertex_normals, farthest_point_sample
from loss import chamfer_dist, arap_cost, compute_truncated_chamfer_distance
from lib.deformation_graph import DeformationGraph_geod

def read_file(file_path):
    lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            try:
                number = int(line)
                lines.append(number)
            except ValueError:
                print(f"Invalid line: {line}")
    return lines

parser = argparse.ArgumentParser(description="Launch the eval of DQFM model.")
parser.add_argument("--config", type=str, default="config", help="Config file name")
args = parser.parse_args()


cfg = yaml.safe_load(open(f"./config/{args.config}.yaml", "r"))
source_temp_name = cfg["source_temp_name"]
source_data_set_name = cfg["source_data_set_name"]
target_data_set_name =  cfg["target_data_set_name"]
model_path =  cfg["model_path"]
save_path = cfg["save_path"]
rot_name = cfg["rot"]
seed = cfg["seed"]
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


if not os.path.exists(os.path.join(save_path,cfg["save_name"])):
    os.makedirs(os.path.join(save_path,cfg["save_name"]))
with open(os.path.join(save_path,cfg["save_name"],'config.yaml'), "w") as f:
        yaml.dump(cfg, f)


if torch.cuda.is_available() and cfg["misc"]["cuda"]:
    device = torch.device(f'cuda:{cfg["misc"]["device"]}')
else:
    device = torch.device("cpu")
base_path = os.path.dirname(__file__)

if cfg["templatename"] == "faust":
    usefaust = True
elif cfg["templatename"] == "shrec19":
    usefaust = 'shrec19tem'    
else:
    usefaust = False

dataset_path_test = os.path.join(cfg["dataset"]["root_dataset"], cfg["dataset"]["root_test"])
dataset_path_train = os.path.join(cfg["dataset"]["root_dataset"], cfg["dataset"]["root_train"])

train_dataset = testDataset(dataset_path_train, name=cfg["source_temp_name"],
                            use_cache=True, train=True,single=True,faustdata = usefaust)
test_dataset = testDataset(dataset_path_test, name=cfg["target_data_set_name"] ,
                            use_cache=True, train=False,single=True,faustdata = cfg["datasetname"])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=None, shuffle=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, shuffle=False)

point_backbone = DecoderSimpleDGCNN_sample(device=device).to(device) 
print(model_path)
point_backbone.load_state_dict(torch.load(model_path, map_location=device))
point_backbone.eval()


with torch.no_grad():
    for i, data in tqdm(enumerate(train_loader)):
        source_mesh = o3d.geometry.TriangleMesh()
        source_mesh_V,source_mesh_F = data["shape1"]['vts'],data["shape1"]['faces']
        source_mesh.vertices =  o3d.utility.Vector3dVector(source_mesh_V)
        source_mesh.triangles = o3d.utility.Vector3iVector(source_mesh_F)
        
        
        name1 = data["shape1"]["name"]
        print('source',name1)
        V1 = data["shape1"]['vts'].unsqueeze(0).to(device)

        V1_sample = data["shape1"]['vts_sample'].unsqueeze(0).to(device)
        FPS1 = data["shape1"]['fps'].unsqueeze(0).to(device)
        V1_fps_sample = FPS1

        dg = DeformationGraph_geod()

        if cfg["misc"]["dgpcl"]:
            geod = compute_geodesic_distmat(source_mesh_V.squeeze(0).cpu().numpy(),source_mesh_F.squeeze(0).cpu().numpy())      
        else:
            geod = torch.cdist(source_mesh_V.squeeze(0), source_mesh_V.squeeze(0), p=2.0).cpu().numpy()
        dg.construct_graph(source_mesh_V,source_mesh_F,geod,device)
        num_nodes = dg.nodes_idx.shape[0]
        break

for i, data in tqdm(enumerate(test_loader)):

    target_pcd = o3d.geometry.PointCloud()
    target_mesh_F = data["shape2"]['faces'].squeeze()       # target_pcd.points = o3d.utility.Vector3dVector(data["shape2"]['vts'])
    target_mesh_V = data["shape2"]['vts'].squeeze()  # 5000 3
    target_forgth = target_mesh_V
    name2 = data["shape2"]["name"]
    print(name1)
    print(name2)

    target_mesh_V_fps = farthest_point_sample(target_mesh_V,target_mesh_V.shape[0]) # 2514
    target_pcd.points = o3d.utility.Vector3dVector(target_mesh_V)

    V2 = target_mesh_V.unsqueeze(0).to(device) # 2514 3
    V2_fps_sample = target_mesh_V_fps[:,:1500].long().to(device) # 1500
    V2_sample = V2[0,V2_fps_sample[0]].unsqueeze(0).to(device) # 1500 3
    with torch.no_grad():

        #feat1, feat2 = point_backbone(V1.permute(0,2,1)), point_backbone(V2.permute(0,2,1))
        feat1, feat2 = point_backbone(V1.permute(0,2,1),V1_sample.permute(0,2,1),V1_fps_sample), point_backbone(V2.permute(0,2,1),V2_sample.permute(0,2,1),V2_fps_sample)    
        T12_pred,T21_pred= search_t(feat1, feat2), search_t(feat2, feat1)

        save_path_t = os.path.join(save_path ,cfg["save_name"]+'_unregister', f'T')
        if not os.path.exists(save_path_t):
            os.makedirs(save_path_t)
        filename_t12 = f'T_{name1}_{name2}.txt'
        t12 = T12_pred.detach().cpu().squeeze(0).numpy()
        np.savetxt(os.path.join(save_path_t, filename_t12), t12, fmt='%i')
        filename_t21 = f'T_{name2}_{name1}.txt'
        t21 = T21_pred.detach().cpu().squeeze(0).numpy()
        np.savetxt(os.path.join(save_path_t, filename_t21), t21, fmt='%i')

    opt_d_rotations = torch.zeros((1, num_nodes, 3)).to(device) # axis angle 
    opt_d_translations = torch.zeros((1, num_nodes, 3)).to(device)
    opt_d_rotations.requires_grad = True
    opt_d_translations.requires_grad = True
    surface_opt_params = [opt_d_rotations, opt_d_translations] 
    surface_optimizer = torch.optim.Adam(surface_opt_params, lr=0.005, betas=(0.9, 0.999))


    source_mesh_V = source_mesh_V.to(device)
    target_mesh_V = target_mesh_V.to(device)




    using_xyz = False
    count = 0
    idx = 0
    eps = 1e-8

    while True:
        if idx > 8000:
            save_path_t = os.path.join(save_path ,cfg["save_name"], f'T')
            if not os.path.exists(save_path_t):
                os.makedirs(save_path_t)
            filename_t12 = f'T_{name1}_{name2}.txt'
            t12 = T12_pred.detach().cpu().squeeze(0).numpy()
            np.savetxt(os.path.join(save_path_t, filename_t12), t12, fmt='%i')
            filename_t21 = f'T_{name2}_{name1}.txt'
            t21 = T21_pred.detach().cpu().squeeze(0).numpy()
            np.savetxt(os.path.join(save_path_t, filename_t21), t21, fmt='%i')


            save_path_mesh = os.path.join(save_path ,cfg["save_name"], f'mesh')
            if not os.path.exists(save_path_mesh):
                os.makedirs(save_path_mesh)
            save_mesh = o3d.geometry.TriangleMesh()
            save_mesh.vertices =  o3d.utility.Vector3dVector(new_source_mesh_V.squeeze().detach().cpu().numpy())
            save_mesh.triangles = o3d.utility.Vector3iVector(source_mesh_F)
            o3d.io.write_triangle_mesh(os.path.join(save_path_mesh,f'{name1}_{name2}.off'), save_mesh)
            print('total iter:', idx)
            break
        
        S_idx = torch.arange(0,source_mesh_V.shape[0]).to(device)
        S_temp1 = S_idx[T21_pred.squeeze() -1 ]
        S_temp2 = S_temp1[T12_pred.squeeze() -1]   
        Geo_dist_idx = geod[S_idx.cpu().numpy(),S_temp2.cpu().numpy()] < 0.001
        Geo_dist_idx = torch.from_numpy(Geo_dist_idx).to(device)
        


        new_source_mesh_V,arap,sr_loss= dg(source_mesh_V,opt_d_rotations, opt_d_translations)
        

        ## new source mesh
        target_mesh_V_T = target_mesh_V.squeeze()[T12_pred.squeeze() -1 ]
        target_mesh_V_nodes = target_mesh_V_T[Geo_dist_idx>0]
        new_source_mesh_V_nodes = new_source_mesh_V[:,Geo_dist_idx>0].squeeze()
    
        ## loss
        cd_loss = chamfer_dist(new_source_mesh_V.squeeze(),target_mesh_V.squeeze())
        loss_ali = (target_mesh_V_nodes - new_source_mesh_V_nodes)**2
        loss_ali =   torch.mean(   loss_ali   )

        if using_xyz is False:
            loss =  cfg["stage1"]["rmse"] * loss_ali + cfg["stage1"]["arap"]* arap + cfg["stage1"]["cd"]*cd_loss + cfg["stage1"]["sr"]* sr_loss

        else:
            loss = cfg["stage2"]["rmse"] * loss_ali + cfg["stage2"]["arap"]* arap + cfg["stage2"]["cd"]*cd_loss + cfg["stage2"]["sr"]* sr_loss

        if idx == 0:
            last_loss = torch.tensor([0]).to(device)
        elif abs(last_loss.item() - loss.item()) > eps:
            
            last_loss = loss.clone()
        else:
            count += 1
            
            if count > 15:

                if using_xyz is False:
                    using_xyz = True
                    count = 0
                    idx_recorder = idx
                    eps = 1e-8
                    print("using xyz",idx)
                else:

                    save_path_t = os.path.join(save_path ,cfg["save_name"], f'T')
                    if not os.path.exists(save_path_t):
                        os.makedirs(save_path_t)
                    filename_t12 = f'T_{name1}_{name2}.txt'
                    t12 = T12_pred.detach().cpu().squeeze(0).numpy()
                    np.savetxt(os.path.join(save_path_t, filename_t12), t12, fmt='%i')
                    filename_t21 = f'T_{name2}_{name1}.txt'
                    t21 = T21_pred.detach().cpu().squeeze(0).numpy()
                    np.savetxt(os.path.join(save_path_t, filename_t21), t21, fmt='%i')


                    save_path_mesh = os.path.join(save_path ,cfg["save_name"], f'mesh')
                    if not os.path.exists(save_path_mesh):
                        os.makedirs(save_path_mesh)
                    save_mesh = o3d.geometry.TriangleMesh()
                    save_mesh.vertices =  o3d.utility.Vector3dVector(new_source_mesh_V.squeeze().detach().cpu().numpy())
                    save_mesh.triangles = o3d.utility.Vector3iVector(source_mesh_F)
                    o3d.io.write_triangle_mesh(os.path.join(save_path_mesh,f'{name1}_{name2}.off'), save_mesh)
                    break


        if idx % 100 == 0:
            with torch.no_grad():
                if using_xyz is False:
                    new_source_mesh_V = new_source_mesh_V.squeeze()
                    V1 = new_source_mesh_V.unsqueeze(0).to(device)
                    V1_sample = new_source_mesh_V[V1_fps_sample.squeeze()].unsqueeze(0).to(device)

                    feat1, feat2 = point_backbone(V1.permute(0,2,1),V1_sample.permute(0,2,1),V1_fps_sample), point_backbone(V2.permute(0,2,1),V2_sample.permute(0,2,1),V2_fps_sample) 
                    T12_pred = search_t(feat1, feat2) 
                    T21_pred = search_t(feat2, feat1) 


                else:

                    T12_pred = search_t(new_source_mesh_V, target_mesh_V) 
                    T21_pred = search_t(target_mesh_V, new_source_mesh_V) 
     
              

        idx = idx + 1

        surface_optimizer.zero_grad()
        loss.backward()
        surface_optimizer.step()
        
