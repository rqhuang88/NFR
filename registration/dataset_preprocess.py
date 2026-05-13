import argparse
import os
import random

import numpy as np
import potpourri3d as pp3d
import open3d as o3d
import torch

from cal_ico import cal_icosahedron
from utils import farthest_point_sample, compute_vertex_normals


def filter_rows(A, B):
    filtered_rows = []
    for row in A:
        if np.all(np.in1d(row, B)):
            filtered_rows.append(row)
    filtered_A = np.array(filtered_rows)
    return filtered_A


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset: generate partial views and save results.")
    parser.add_argument("--data_dir", type=str, default="../data/faust_raw/", help="Path to input mesh directory")
    parser.add_argument("--save_path", type=str, default="../results/data", help="Output save path")
    parser.add_argument("--save_name", type=str, default="faust_raw", help="Output subdirectory name")
    args = parser.parse_args()

    save_path = args.save_path
    save_name = args.save_name

    for root, dirs, files in os.walk(args.data_dir):
        for file in files:
            pth = os.path.join(root, file)
            verts, faces = pp3d.read_mesh(pth)
            shape_name = os.path.splitext(file)[0]
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))
            target_normal = compute_vertex_normals(verts, faces)

            rotation_matrix = cal_icosahedron()
            i = random.randint(0, 11)
            rotated_normal = torch.matmul(target_normal, torch.tensor(rotation_matrix[i]).float())
            idx_partial = torch.from_numpy(np.asarray(np.where(rotated_normal[:, 2] > 0))).long().squeeze()
            verts_pc = verts[idx_partial]
            fps = farthest_point_sample(verts_pc, verts_pc.shape[0]).squeeze()
            fps = fps[:3000]
            verts_pc = verts_pc[fps]
            idx = idx_partial[fps].detach().cpu().squeeze(0).numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(verts_pc)
            filtered_face = filter_rows(faces.numpy(), idx)
            for pair in filtered_face:
                pair[0] = np.where(idx == pair[0])[0][0]
                pair[1] = np.where(idx == pair[1])[0][0]
                pair[2] = np.where(idx == pair[2])[0][0]

            filtered_face = torch.tensor(filtered_face)
            save_path_t = os.path.join(save_path, save_name, 'index_partial')
            if not os.path.exists(save_path_t):
                os.makedirs(save_path_t)
            filename_index = f'index_{shape_name}.txt'
            idx1p = idx_partial.detach().cpu().squeeze(0).numpy()
            np.savetxt(os.path.join(save_path_t, filename_index), idx1p, fmt='%i')

            save_path_t = os.path.join(save_path, save_name, 'points')
            if not os.path.exists(save_path_t):
                os.makedirs(save_path_t)
            filename_pcd = f'pcd_{shape_name}.ply'
            o3d.io.write_point_cloud(os.path.join(save_path_t, filename_pcd), pcd)

            save_path_mesh = os.path.join(save_path, save_name, 'mesh')
            if not os.path.exists(save_path_mesh):
                os.makedirs(save_path_mesh)
            save_mesh = o3d.geometry.TriangleMesh()
            save_mesh.vertices = o3d.utility.Vector3dVector(verts_pc.squeeze().detach().cpu().numpy())
            save_mesh.triangles = o3d.utility.Vector3iVector(filtered_face)
            o3d.io.write_triangle_mesh(os.path.join(save_path_mesh, f'{shape_name}.off'), save_mesh)
