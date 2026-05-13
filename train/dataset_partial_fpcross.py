import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import potpourri3d as pp3d
import scipy.io as scio
import torch
import yaml
from torch.utils.data import Dataset
import open3d as o3d
from tqdm import tqdm
from itertools import permutations
from partial import fullpoint_to_partial, compute_vertex_normals
import diffusion_net as dfn
from utils import farthest_point_sample, pc_normalize, search_t
from cal_ico import cal_icosahedron

# change:read txt
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

def cal_geo(V,F):
    dist = torch.tensor([])
    for i in range(V.shape[0]):
        dist = torch.cat([dist,torch.tensor(pp3d.compute_distance(V,F,i)).unsqueeze(1)],dim=-1)
    # [][to point]
    return dist


def tosca_pairs(split):
    catlength = 10
    centaurlength = 6
    davidlength = 7
    doglength = 9 
    horselength = 8 
    michaellength = 20
    victorialength = 12
    wolflength = 3
    if split == 'train':
        tag = 0
    else:
        tag = 1
    combination1 = split_data(list(permutations(range(catlength+1), 2)))[tag]
    combination2 = split_data(list(permutations(range(catlength+1,catlength+1+centaurlength), 2)))[tag]
    combination3 = split_data(list(permutations(range(catlength+1+centaurlength,catlength+1+centaurlength+davidlength), 2)))[tag]
    combination4 = split_data(list(permutations(range(catlength+1+centaurlength+davidlength,catlength+1+centaurlength+davidlength+doglength), 2)))[tag]
    combination5 = split_data(list(permutations(range(catlength+1+centaurlength+davidlength+doglength,catlength+1+centaurlength+davidlength+doglength+horselength), 2)))[tag]
    combination6 = split_data(list(permutations(range(catlength+1+centaurlength+davidlength+doglength+horselength,catlength+1+centaurlength+davidlength+doglength+horselength+michaellength), 2)))[tag]
    combination7 = split_data(list(permutations(range(catlength+1+centaurlength+davidlength+doglength+horselength+michaellength,catlength+1+centaurlength+davidlength+doglength+horselength+michaellength+victorialength), 2)))[tag]
    combination8 = split_data(list(permutations(range(catlength+1+centaurlength+davidlength+doglength+horselength+michaellength+victorialength,catlength+1+centaurlength+davidlength+doglength+horselength+michaellength+victorialength+wolflength), 2)))[tag]
    return (combination1+combination2+combination3+combination4+combination5+combination6+combination7+combination8)

def split_data(array):

    split_length = 4*len(array) // 5

    train_split = array[:split_length]
    test_split = array[split_length:]

    return [train_split,test_split]

def split_and_save(array):

    half_length = 4*len(array) // 5

    first_half = array[:half_length]
    second_half = array[half_length:]

    with open('train_pairs.txt', 'a') as file:
        for item in first_half:
            file.write(f'{item[0]}\t{item[1]}\n')

    with open('test_pairs.txt', 'a') as file:
        for item in second_half:
            file.write(f'{item[0]}\t{item[1]}\n')


class DFRDataset(Dataset):

    def __init__(self, root_dir, name="remeshed",
                 with_wks=None, with_sym=False,
                 use_cache=True, op_cache_dir=None, 
                 class_name='faust', train=True,cfg=None):

        self.root_dir = root_dir
        self.cache_dir = root_dir
        self.op_cache_dir = op_cache_dir
        self.with_sym = with_sym
        self.cfg= cfg
        # check the cache
        split = "train" if train else "test"
        wks_suf = "" if with_wks is None else "wks_"
        sym_suf = "" if not with_sym else "sym_"

        # check the cache
        split = "train" if train else "test"
        if use_cache:
            load_cache = os.path.join(self.cache_dir, f"cache_{name}_{split}.pt")
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                (
                    # main
                    self.verts_list,
                    self.massvec_list,
                    self.faces_list,
                    self.phi_list,
                    self.eval_list,
                    self.used_shapes,
                    self.vts_list,
                    self.fps_list,
                    self.dist_list,
                    self.normals_list
                ) = torch.load(load_cache)
                self.combinations = list(permutations(range(len(self.verts_list)), 2))
                return
            print("  --> dataset not in cache, repopulating")

        # Load the meshes
        # define files and order
        shapes_split = "shapes_" + split
        self.used_shapes = sorted([x.stem for x in (Path(root_dir) / shapes_split).iterdir() if 'DS_' not in x.stem])
        if self.cfg["datasetname"] == "TOSCA":
            self.combinations = tosca_pairs(split)
        else:
            self.combinations = list(permutations(range(len(self.used_shapes)), 2))
        self.root_dir = root_dir
        result = []
        for start, end in self.combinations:
            result.append((self.used_shapes[start], self.used_shapes[end]))
        print(result)
        mesh_dirpath = Path(root_dir) / shapes_split
        fps_dirpath  = Path(root_dir) / "FPS"
        extfps = '.npy'
        # Get all the files
        ext = '.off'
        self.verts_list = []
        self.faces_list = []
        self.vts_list = []
        self.fps_list = []
        self.normals_list = []
        self.dist_list = []
        # Load the actual files
        for shape_name in tqdm(self.used_shapes):
            
            verts, faces = pp3d.read_mesh(str(mesh_dirpath / f"{shape_name}{ext}"))  # off ob

            print('Cal Geo..')
            dist = cal_geo(verts,faces)

            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            #vts = tuple(i for i in range(0, verts.size()[0]))
            faces = torch.tensor(np.ascontiguousarray(faces))
            noraml = compute_vertex_normals(verts, faces)
            noraml = torch.tensor(np.ascontiguousarray(noraml)).float()
            if ('faust_fps' in  self.root_dir) or ('scape_fps' in self.root_dir):
                if len(shape_name) > 7:
                    fps_name = str(int(shape_name[7:10]))
                else:
                    fps_name = str(int(shape_name[4:7]))
                
                fps = torch.tensor(np.load(str(fps_dirpath / f"{fps_name}{extfps}"))).long()[0].squeeze()
            else:
                fps = farthest_point_sample(verts,verts.shape[0]).squeeze()


            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.fps_list.append(fps)
            self.normals_list.append(noraml)
            self.dist_list.append(dist)
            # vts
            # vts = torch.tensor(np.ascontiguousarray(verts)).float()
            # with vts
            if self.cfg["datasetname"] == "scape":
                vts_dirpath = Path(root_dir) / "corres"
                vts = np.loadtxt(os.path.join(vts_dirpath, f'{shape_name}.vts'), dtype=int) - 1
                self.vts_list.append(vts)

        # Precompute operators
        (
            self.frames_list,
            self.massvec_list,
            self.L_list,
            self.eval_list,
            self.phi_list,
            self.gradX_list,
            self.gradY_list,
        ) = dfn.geometry.get_all_operators(
            self.verts_list,
            self.faces_list,
            k_eig=128,
            op_cache_dir=self.op_cache_dir,
        )
        print('done')

        # save to cache
        if use_cache:
            ensure_dir_exists(self.cache_dir)
            torch.save(
                (
                    self.verts_list,
                    self.massvec_list,
                    self.faces_list,
                    self.phi_list,
                    self.eval_list,
                    self.used_shapes,
                    self.vts_list,
                    self.fps_list,
                    self.dist_list,
                    self.normals_list
                ),
                load_cache,
            )

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):

        # get indexes
        idx1, idx2 = self.combinations[idx]
        self.verts_list[idx1] = pc_normalize(self.verts_list[idx1])
        self.verts_list[idx2] = pc_normalize(self.verts_list[idx2])


        # normal1 = self.normals_list[idx1]


        while True:
            random_integer = random.randint(0, 11)
            # rotation_matrix = cal_icosahedron()
            # rotated_normal = torch.matmul(normal1, torch.tensor(rotation_matrix[random_integer]).float())
            # idx_partial = torch.from_numpy(np.asarray(np.where(rotated_normal[:,2]>0))).long().squeeze()
            partial_path = os.path.join(self.cfg["dataset"]["root_dataset"], self.cfg["datasetname"],'index_partial')
            pth = partial_path + "/index_" + self.used_shapes[idx1] +"_view_"+ str(random_integer+1)+ ".txt"  
            #print(pth)
            idx_partial = torch.tensor(read_file(pth)).long().squeeze()
            if idx_partial.shape[0] > 2200:
                break

        # print(idx_partial.shape)
        verts_1 = self.verts_list[idx1][idx_partial]
        phi_1 = self.phi_list[idx1][idx_partial]
        # print(vts_1.shape)
        # exit()
        fps_1 = farthest_point_sample(verts_1, verts_1.shape[0]).squeeze()
        fps_1 = fps_1[:2200]  

        fps_1_sample = torch.arange(0,1500).long()

        
              
        verts1 = verts_1[fps_1]

        verts1_sample = verts_1[fps_1[:1500]]

        phi_1 = phi_1[fps_1]

        
        fps2 = self.fps_list[idx2][:4995]

        fps_2_sample = torch.arange(0,3000).long()
        verts2_sample = self.verts_list[idx2][fps2[:3000]]
        verts2 = self.verts_list[idx2][fps2]

        fps1 = self.fps_list[idx1][:4995]
        verts1_full = self.verts_list[idx1][fps1]
        # print(idx1)
        shape1 = {
            "verts":verts1,
            "verts_full":verts1_full,
            "dist":self.dist_list[idx1][fps1][:,fps1],
            "phi": 1,
            "phi_partial": phi_1,
            "mass": self.massvec_list[idx1][fps1],
            "phi_full":self.phi_list[idx1][fps1],
            "eval": self.eval_list[idx1],
            "name": self.used_shapes[idx1],
            "verts_sample":verts1_sample,
            "fps_sample": fps_1_sample
        }

        shape2 = {
            "verts":verts2,
            "verts_full":verts2,
            "dist":self.dist_list[idx2][fps2][:,fps2],
            "phi": self.phi_list[idx2][fps2],
            "phi_partial": 1,
            "mass": self.massvec_list[idx2][fps2],
            "phi_full":self.phi_list[idx2][fps2],
            "eval": self.eval_list[idx2],
            "name":self.used_shapes[idx2],
            "verts_sample":verts2_sample,
            "fps_sample": fps_2_sample
        }
        # print(self.phi_list)
        # name1 = self.used_shapes[idx1]
        # name2 = self.used_shapes[idx2]
        # device = 'cuda:0'
        # evec_1, evec_2 = self.phi_list[idx1][:, :50].to(device), self.phi_list[idx2][:, :50].to(device)
        # V1_full = self.verts_list[idx1].unsqueeze(0).to(device)
        # V2_full = self.verts_list[idx2].unsqueeze(0).to(device)
        # point_backbone = self.point_backbone
        # with torch.no_grad():
        #     point_backbone.eval()
        #     feat1, feat2 = point_backbone(V1_full.permute(0,2,1)), point_backbone(V2_full.permute(0,2,1))
        #     T21,T12= search_t(feat1, feat2).squeeze(0).squeeze(1)-1, search_t(feat2, feat1).squeeze(0).squeeze(1)-1
        #     Pi12, Pi21 = T2Pi(T12,T21)
        #     C12_gt = torch.pinverse(evec_2) @ Pi12 @ evec_1
        #     C21_gt = torch.pinverse(evec_1) @ Pi21 @ evec_2

 
        mask12 = torch.ones([10, 10])
        mask21 = torch.ones([10, 10])
        return {"shape1": shape1, "shape2": shape2, 'mask12': mask12, 'mask21': mask21}



class DFRDataset_eval(Dataset):

    def __init__(self, root_dir, name="remeshed_test",
                 with_wks=None, with_sym=False,
                 use_cache=True, op_cache_dir=None, 
                 class_name='faust', train=True,cfg=None,faustdata = False):

        self.root_dir = root_dir
        self.cache_dir = root_dir
        self.op_cache_dir = op_cache_dir
        self.with_sym = with_sym
        self.cfg= cfg
        self.split = "train" if train else "test"
        # check the cache
        split = "train" if train else "test"
        wks_suf = "" if with_wks is None else "wks_"
        sym_suf = "" if not with_sym else "sym_"

        # check the cache
        split = "train" if train else "test"
        if use_cache:
            load_cache = os.path.join(self.cache_dir, f"cache_{name}_{split}.pt")
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                (
                    # main
                    self.verts_list,
                    self.faces_list,
                    self.phi_list,
                    self.eval_list,
                    self.used_shapes,
                    self.vts_list,
                    self.fps_list,
                    self.normals_list
                ) = torch.load(load_cache)

                if faustdata == True:
                    self.combinations = list(permutations(range(69,len(self.used_shapes)), 2))
                elif split  == "train":
                    # numb = 0
                # self.combinations = list(permutations(range(len(self.verts_list)), 2))[numb*(len(self.verts_list)-1):numb*(len(self.verts_list)-1)+len(self.verts_list)-1]
                    self.combinations = [(0,0)]
                elif self.cfg["test"]["test_full2full"]:
                    self.combinations = list(permutations(range(len(self.verts_list)), 2))
                else:
                    numb = 0
                    self.combinations =  []
                    for i in range(len(self.verts_list)):
                       self.combinations.append((i,0))
                return
            print("  --> dataset not in cache, repopulating")

        # Load the meshes
        # define files and order
        shapes_split = "shapes_" + split
        self.used_shapes = sorted([x.stem for x in (Path(root_dir) / shapes_split).iterdir() if 'DS_' not in x.stem])
        self.combinations = list(permutations(range(len(self.used_shapes)), 2))
        self.root_dir = root_dir

        mesh_dirpath = Path(root_dir) / shapes_split
        fps_dirpath  = Path(root_dir) / "FPS"
        vts_dirpath = Path(root_dir) / "corres"
        extfps = '.npy'
        # Get all the files
        ext = '.off'
        self.verts_list = []
        self.faces_list = []
        self.vts_list = []
        self.fps_list = []
        self.normals_list = []
        # Load the actual files
        for shape_name in tqdm(self.used_shapes):
            print(mesh_dirpath)
            print(shape_name)
            verts, faces = pp3d.read_mesh(str(mesh_dirpath / f"{shape_name}{ext}"))  # off ob

            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))
            noraml = compute_vertex_normals(verts, faces)
            noraml = torch.tensor(np.ascontiguousarray(noraml)).float()
            if ('faust_fps' in  self.root_dir) or ('scape_fps' in self.root_dir):
                if len(shape_name) > 7:
                    fps_name = str(int(shape_name[7:10]))
                else:
                    fps_name = str(int(shape_name[4:7]))
                
                fps = torch.tensor(np.load(str(fps_dirpath / f"{fps_name}{extfps}"))).long()[0].squeeze()
            else:
                fps = farthest_point_sample(verts,verts.shape[0]).squeeze()


            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.fps_list.append(fps)
            self.normals_list.append(noraml)
            # vts
            # vts = torch.tensor(np.ascontiguousarray(verts)).float()
            # vts = np.loadtxt(os.path.join(vts_dirpath, f'{shape_name}.vts'), dtype=int) - 1
            # self.vts_list.append(vts)

        # Precompute operators
        (
            self.frames_list,
            self.massvec_list,
            self.L_list,
            self.eval_list,
            self.phi_list,
            self.gradX_list,
            self.gradY_list,
        ) = dfn.geometry.get_all_operators(
            self.verts_list,
            self.faces_list,
            k_eig=128,
            op_cache_dir=self.op_cache_dir,
        )
        print('done')

        # save to cache
        if use_cache:
            ensure_dir_exists(self.cache_dir)
            torch.save(
                (
                    self.verts_list,
                    self.faces_list,
                    self.phi_list,
                    self.eval_list,
                    self.used_shapes,
                    self.vts_list,
                    self.fps_list,
                    self.normals_list
                ),
                load_cache,
            )

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):

        # get indexes
        idx1, idx2 = self.combinations[idx]
        self.verts_list[idx1] = pc_normalize(self.verts_list[idx1])
        self.verts_list[idx2] = pc_normalize(self.verts_list[idx2])


        #normal1 = self.normals_list[idx1]



        # while True:
        #     angle = 360*random.random()
        #     radian = torch.deg2rad(torch.tensor(angle))

        #     rotation_matrix = torch.tensor([[torch.cos(radian), 0.0, torch.sin(radian)],
        #                 [0.0, 1.0, 0.0],
        #                 [-torch.sin(radian), 0.0, torch.cos(radian)]], dtype=torch.float32)
        #     rotated_normal = torch.matmul(normal1, rotation_matrix)
        #     idx_partial = torch.from_numpy(np.asarray(np.where(rotated_normal[:,2]>0))).long().squeeze()
        #     if idx_partial.shape[0] > 2200:
        #         break
        if self.split == 'test':
            name = self.used_shapes[idx1]
            cfg = self.cfg
            dataset_path_test = os.path.join(cfg["dataset"]["root_dataset"], cfg["dataset"]["root_test"])
            partial_path = os.path.join(dataset_path_test, cfg["indextype"])
            pth = partial_path + "/index_" + name +".txt"  
            print(pth)
            idx_partial = torch.tensor(read_file(pth)).long().squeeze()
        else:
            normal1 = self.normals_list[idx1]

            while True:
                angle = 360*random.random()
                radian = torch.deg2rad(torch.tensor(angle))

                rotation_matrix = torch.tensor([[torch.cos(radian), 0.0, torch.sin(radian)],
                            [0.0, 1.0, 0.0],
                            [-torch.sin(radian), 0.0, torch.cos(radian)]], dtype=torch.float32)
                rotated_normal = torch.matmul(normal1, rotation_matrix)
                idx_partial = torch.from_numpy(np.asarray(np.where(rotated_normal[:,2]>0))).long().squeeze()
                if idx_partial.shape[0] > 2200:
                    break    
        #print(idx_partial.shape)
        #verts1 = self.verts_list[idx1][idx_partial]
        # full
        verts1 = self.verts_list[idx1]
        # phi_1 = self.phi_list[idx1][idx_partial]
        # print(vts_1.shape)
        # exit()
        verts2 = self.verts_list[idx2]

        fps_1 = farthest_point_sample(verts1, verts1.shape[0]).squeeze()
        fps2 = self.fps_list[idx2]

        fps_1_sample = fps_1[:1500].long()
        fps_2_sample = fps2[:3000].long()

        verts1_sample = verts1[fps_1[:1500]]
        verts2_sample = verts2[fps2[:3000]]
        shape1 = {
            "verts":verts1,
            "idx_partial": idx_partial,
            "name": self.used_shapes[idx1],
            "verts_sample":verts1_sample,
            "fps_sample": fps_1_sample
        }

        shape2 = {
            "verts":verts2,
            "name":self.used_shapes[idx2],
            "verts_sample":verts2_sample,
            "fps_sample": fps_2_sample
        }

        return {"shape1": shape1, "shape2": shape2}

def shape_to_device(dict_shape, device):
    names_to_device = ["verts","verts_full","dist","mass","phi_partial","phi_full","eval","phi","verts_sample","fps_sample"]
    for k, v in dict_shape.items():
        if "shape" in k:
            for name in names_to_device:
                # if v[name] is not None:
                if name in v.keys():
                    v[name] = v[name].to(device)  # .float()
            dict_shape[k] = v
        else:
            dict_shape[k] = v.to(device)

    return dict_shape
def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)

def T2Pi(T12_init, T21_init):
    device = T12_init.device
    n1, n2 = T12_init.shape[0], T21_init.shape[0]
    T12, T21 = torch.zeros(n1, n2).to(device), torch.zeros(n2, n1).to(device)
    L1= torch.arange(n1).to(device)
    L2 = torch.arange(n2).to(device)
    T12[L1,T12_init] = 1
    T21[L2,T21_init] = 1
    # for i in range(n1):
    #     T12[i, T12_init[i]] = 1
    # for j in range(n2):
    #     T21[j, T21_init[j]] = 1
    return T12, T21

# class DFRDataset(Dataset):
#     def __init__(self, root_dir, name="scape-remeshed",
#                  with_wks=None, with_sym=False,
#                  use_cache=True, op_cache_dir=None, 
#                  class_name='faust', train=True):

#         self.root_dir = root_dir
#         self.cache_dir = root_dir
#         self.op_cache_dir = op_cache_dir
#         self.with_sym = with_sym
#         # check the cache
#         split = "train" if train else "test"
#         wks_suf = "" if with_wks is None else "wks_"
#         sym_suf = "" if not with_sym else "sym_"
#         if use_cache:
#             load_cache = os.path.join(self.cache_dir, f"cache_{name}_{sym_suf}{wks_suf}{split}.pt")
#             print("using dataset cache path: " + str(load_cache))
#             if os.path.exists(load_cache):
#                 print("  --> loading dataset from cache")
#                 (
#                     self.verts_list,
#                     self.f_list,
#                     self.phi_list,
#                     # self.phi_inv_list,
#                     self.fps_list,
#                     self.eval_list,
#                     self.vts_list,
#                     self.verts_partial_list,
#                     self.fps_partial_list,
#                     self.idx_partial_list,
#                     self.used_shapes,
#                 ) = torch.load(load_cache)

#                 self.combinations = list(permutations(range(len(self.verts_list)), 2))
#                 return
#             print("  --> dataset not in cache, repopulating")

#         # Load the meshes
#         # define files and order
#         shapes_split = "shapes_" + split
#         self.used_shapes = sorted([x.stem for x in (Path(root_dir) / shapes_split).iterdir() if 'DS_' not in x.stem])
#         print(self.used_shapes)
#         # set combinations
#         self.combinations = list(permutations(range(len(self.used_shapes)), 2))

#         #
#         mesh_dirpath = Path(root_dir) / shapes_split
#         vts_dirpath = Path(root_dir) / "corres"

#         # Get all the files
#         ext = '.off'
#         self.verts_list = []
#         self.faces_list = []
#         self.f_list = []
#         self.phi_list = []
#         self.phi_inv_list = []
#         self.fps_list = []
#         self.eval_list = []
#         self.vts_list = []
#         self.verts_partial_list = []
#         self.fps_partial_list = []
#         self.idx_partial_list = []
        
#         # Load the actual files
#         for shape_name in tqdm(self.used_shapes):
#             i = 0
#             data_dir = './mesh_results/' + class_name + '/'
#             # feature_name = 'F/F_' + f'{shape_name}.mat'
#             # Phi_name = 'Phi/Phi_' + f'{shape_name}.mat'
#             # Phi_inv_name = 'Phi_inv/Phi_inv_' + f'{shape_name}.mat'
#             # Eval_name = 'Eval/Eval_' + f'{shape_name}.mat'

#             try:
#                 verts, faces = pp3d.read_mesh(str(mesh_dirpath / f"{shape_name}{ext}"))
#                 # feature = scio.loadmat(os.path.join(data_dir, feature_name))['F']
#                 # Phi = scio.loadmat(os.path.join(data_dir, Phi_name))['Phi']
#                 # Phi_inv = scio.loadmat(os.path.join(data_dir, Phi_inv_name))['Phi_inv']
#                 # Eval = scio.loadmat(os.path.join(data_dir, Eval_name))['Eval']
#                 vts = np.loadtxt(os.path.join(vts_dirpath, f'{shape_name}.vts'), dtype=int) - 1
#             except:
#                 exit()

#             # to torch
#             verts = torch.tensor(np.ascontiguousarray(verts)).float()
#             # feature = torch.tensor(np.ascontiguousarray(feature)).float()
#             faces = torch.tensor(np.ascontiguousarray(faces))

#             fps = farthest_point_sample(verts, verts.shape[0]).squeeze()
            
#             ##verts_partial = verts[normal[:,2]>0].float() # 2514 3
#             ##idx_partial = torch.from_numpy(np.asarray(np.where(normal[:,2]>0))).float()

#             rotated_vertices = verts
#             idx_partial = None
#             angle = 360*random.random()
#             radian = torch.deg2rad(torch.tensor(angle))

#             rotation_matrix = torch.tensor([[torch.cos(radian), 0.0, torch.sin(radian)],
#                                 [0.0, 1.0, 0.0],
#                                 [-torch.sin(radian), 0.0, torch.cos(radian)]], dtype=torch.float32)

#             rotated_vertices = torch.matmul(rotated_vertices, rotation_matrix)
#             # origin
#             normal = compute_vertex_normals(rotated_vertices, faces)
#             #rotated_vertices = rotated_vertices[normal[:,2]>0].float() # 2514 3
#             idx_partial = torch.from_numpy(np.asarray(np.where(normal[:,2]>0))).float()

#             idx_partial = idx_partial.long().squeeze()[:2000]
#             verts_partial = verts[idx_partial]
#             #print(idx_partial.shape)
            
#             fps_partial = farthest_point_sample(verts_partial, verts_partial.shape[0]).squeeze()
            
#             self.faces_list.append(faces)
#             self.verts_list.append(verts)
#             # self.f_list.append(feature)
#             self.fps_list.append(fps)
#             # self.phi_list.append(Phi)
#             # self.phi_inv_list.append(Phi_inv)
#             # self.eval_list.append(Eval)
#             self.vts_list.append(vts)
#             self.verts_partial_list.append(verts_partial)
#             self.fps_partial_list.append(fps_partial)
#             self.idx_partial_list.append(idx_partial)
        
#         # Precompute operators
#         (
#             self.frames_list,
#             self.massvec_list,
#             self.L_list,
#             self.eval_list,
#             self.phi_list,
#             self.gradX_list,
#             self.gradY_list,
#         ) = dfn.geometry.get_all_operators(
#             self.verts_list,
#             self.faces_list,
#             k_eig=128,
#             op_cache_dir=self.op_cache_dir,
#         )

            

#         # save to cache
#         if use_cache:
#             torch.save(
#                 (
#                     self.verts_list,
#                     self.f_list,
#                     self.phi_list,
#                     self.fps_list,
#                     self.eval_list,
#                     self.vts_list,
#                     self.verts_partial_list,
#                     self.fps_partial_list,
#                     self.idx_partial_list,
#                     self.used_shapes,
#                 ),
#                 load_cache,
#             )

#     def __len__(self):
#         return len(self.combinations)

#     def __getitem__(self, idx):
#         # get indexes
#         idx1, idx2 = self.combinations[idx]
#         idx1_partial, idx2_partial = self.idx_partial_list[idx1].long(), self.idx_partial_list[idx2].long()
#         Phi1_partial, Phi2_partial = self.phi_list[idx1][idx1_partial].squeeze(), self.phi_list[idx2][idx2_partial].squeeze()
#         name1, name2 = self.used_shapes[idx1], self.used_shapes[idx2]
        
#         fps1 = self.fps_list[idx1][:4995]
#         fps2 = self.fps_list[idx2][:4995]        
#         verts1 = self.verts_list[idx1][fps1]
#         verts2 = self.verts_list[idx2][fps2]
     
#         # scape
#         fps1_partial = self.fps_partial_list[idx1][:2000]
#         fps2_partial = self.fps_partial_list[idx2][:2000]
#         verts1_partial = self.verts_partial_list[idx1][fps1_partial]
#         verts2_partial = self.verts_partial_list[idx2][fps2_partial]
   
        
#         shape1 = {
#             "verts":verts1,
#             # "verts_smaple":verts1_sample,
#             # "fps": fps1_sample,
#             "verts_partial":verts1_partial,
#             "phi": self.phi_list[idx1][fps1],
#             "phi_partial": Phi1_partial[fps1_partial],
#             # "phi_inv": self.phi_inv_list[idx1][:,fps1],
#             "eval": self.eval_list[idx1],
#             "idx": fps1_partial,
#             "name": name1,
#         }

#         shape2 = {
#             "verts":verts2,
#             # "verts_smaple":verts2_sample,
#             # "fps": fps2_sample,
#             "verts_partial":verts2_partial,
#             # "verts_partial_smaple":verts2_partial_sample,
#             # "fps_partial": fps2_partial_sample,
#             # "feature": self.f_list[idx2][fps2],
#             # "phi": self.phi_list[idx2][fps2],
#             "phi": self.phi_list[idx2][fps2],
#             "phi_partial": Phi2_partial[fps2_partial],
#             # "phi_inv": self.phi_inv_list[idx2][:,fps2],
#             "eval": self.eval_list[idx2],
#             "idx": fps2_partial,
#             "name": name2,
#         }
        
#         # Compute fmap
#         evec_1, evec_2 = self.phi_list[idx1][:, :50], self.phi_list[idx2][:, :50]
#         vts1, vts2 = self.vts_list[idx1], self.vts_list[idx2]


#         un =  False
#         if un is True:

#             T21_name = 'T_' + name2 + '_' + name1 + '.txt'
#             T12_name = 'T_' + name1 + '_' + name2 + '.txt'
#             C12_gt = torch.pinverse(evec_2) @ evec_1[T12]
#             C21_gt = torch.pinverse(evec_1) @ evec_2[T21]


#             # C12_gt_1 = torch.pinverse(evec_2[vts2]) @ evec_1[vts1]
#             # C21_gt_1 = torch.pinverse(evec_1[vts1]) @ evec_2[vts2]
#             # print(torch.norm(C12_gt-C12_gt_1)/torch.norm(C12_gt_1),torch.norm(C21_gt-C21_gt_1)/torch.norm(C21_gt_1))
#         else:
#             C12_gt = torch.pinverse(evec_2[vts2]) @ evec_1[vts1]
#             C21_gt = torch.pinverse(evec_1[vts1]) @ evec_2[vts2]

 
#         mask12 = torch.ones([10, 10])
#         mask21 = torch.ones([10, 10])
#         return {"shape1": shape1, "shape2": shape2, "C12_gt": C12_gt, "C21_gt": C21_gt, 'mask12': mask12, 'mask21': mask21}
