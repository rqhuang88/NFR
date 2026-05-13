import os
from pathlib import Path
import numpy as np
import potpourri3d as pp3d
import torch
from torch.utils.data import Dataset
from utils import farthest_point_sample,pc_normalize,compute_vertex_normals
import trimesh
from tqdm import tqdm
from itertools import permutations
from cal_ico import cal_icosahedron
import random

class testDataset(Dataset):

    def __init__(self, root_dir, name="remeshed",use_cache=True,train=True,single=False,faustdata = False):


        self.root_dir = root_dir
        self.cache_dir = root_dir
        self.name = name

        # check the cache
        split = "train" if train else "test"
        self.split = split
        
        if use_cache:
            load_cache = os.path.join(self.cache_dir, f"cache_{name}_{split}.pt")
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                (
                    self.faces_list,
                    self.used_shapes,
                    self.vts_list,
                    self.fps_list,
                    self.uv_list
                ) = torch.load(load_cache)
                numb = 0
                self.combinations = list(permutations(range(len(self.vts_list)), 2))[numb*(len(self.vts_list)-1):numb*(len(self.vts_list)-1)+len(self.vts_list)-1]
                self.combinations.insert(0, (0,0))
                print(self.combinations)
                return
            print("  --> dataset not in cache, repopulating")

        # Load the meshes
        # define files and order
        shapes_split = "shapes_" + split
        self.used_shapes = sorted([x.stem for x in (Path(root_dir) / shapes_split).iterdir() if 'DS_' not in x.stem])
        self.root_dir = root_dir
        #
        mesh_dirpath = Path(root_dir) / shapes_split
        # Get all the files
        extobj = '.obj'
        self.faces_list = []
        self.vts_list = []
        self.fps_list = []
        self.uv_list = []
        # Load the actual files
        for shape_name in tqdm(self.used_shapes):
            mesh = trimesh.load(str(mesh_dirpath / f"{shape_name}{extobj}"))
            verts = mesh.vertices
            faces = mesh.faces
            uv = mesh.visual.uv
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))
            uv = torch.tensor(np.ascontiguousarray(uv))

            if self.split == 'test':
                target_normal = compute_vertex_normals(verts,faces)
                rotation_matrix = cal_icosahedron()
                number = [0,1,2,3]
                i = random.choice(number)
                rotated_normal = torch.matmul(target_normal, torch.tensor(rotation_matrix[i]).float())
                idx_partial = torch.from_numpy(np.asarray(np.where(rotated_normal[:,2]>0))).long().squeeze()
                verts = verts[idx_partial]
                uv = uv[idx_partial]
            else:
                pass
            
            fps = farthest_point_sample(verts,verts.shape[0]).squeeze()

            self.uv_list.append(uv)
            self.faces_list.append(faces)
            self.fps_list.append(fps)
            self.vts_list.append(verts)


        print('done')

        # save to cache
        if use_cache:
            ensure_dir_exists(self.cache_dir)
            torch.save(
                (
                    self.faces_list,
                    self.used_shapes,
                    self.vts_list,
                    self.fps_list,
                    self.uv_list
                ),
                load_cache,
            )

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):

        if self.split == 'test':
            idx1, idx2 = self.combinations[idx]
            fps1 = self.fps_list[idx1]
            fps2 = self.fps_list[idx2]       
            fps1_sample = fps1[:1500]#original:3000
            fps2_sample = fps2[:1500]
            fps1_base = fps1[:3000]#original:3000
            fps2_base = fps2[:3000]
            shape1 = {
                "faces": self.faces_list[idx1],
                "vts": self.vts_list[idx1][fps1_base],
                "vts_sample": self.vts_list[idx1][fps1_sample],
                "name": self.used_shapes[idx1],
                "fps": fps1,
                "uv": self.uv_list[idx1][fps1_base]
            }

            shape2 = {
                # faces-wrong
                "faces": self.faces_list[idx2],
                "vts":self.vts_list[idx2][fps2_base],
                "vts_sample": self.vts_list[idx2][fps2_sample],
                "name": self.used_shapes[idx2],
                "fps": fps2,
                "uv": self.uv_list[idx2][fps2_base]
            }
        else:
        # get indexes
            idx1, idx2 = self.combinations[idx]
            fps1 = self.fps_list[idx1]
            fps2 = self.fps_list[idx2]       
            fps1_sample = fps1[:3000]#original:3000
            fps2_sample = fps2[:3000]
            fps1_base = fps1[:5000]#original:3000
            fps2_base = fps2[:5000]
            shape1 = {
                "faces": self.faces_list[idx1],
                "vts": self.vts_list[idx1][fps1_base],
                "vts_sample": self.vts_list[idx1][fps1_sample],
                "name": self.used_shapes[idx1],
                "fps": fps1,
                "uv": self.uv_list[idx1][fps1_base]
            }

            shape2 = {
                # faces-wrong
                "faces": self.faces_list[idx2],
                "vts":self.vts_list[idx2][fps2_base],
                "vts_sample": self.vts_list[idx2][fps2_sample],
                "name": self.used_shapes[idx2],
                "fps": fps2,
                "uv": self.uv_list[idx2][fps2_base]
            }
        return {"shape1": shape1, "shape2": shape2}


def shape_to_device(dict_shape, device):
    names_to_device = ["vts", "vts_sample","faces"]
    for k, v in dict_shape.items():
        if "shape" in k:
            for name in names_to_device:
                if v[name] is not None:
                    v[name] = v[name].to(device)  # .float()
            dict_shape[k] = v
        else:
            dict_shape[k] = v.to(device)

    return dict_shape
def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)