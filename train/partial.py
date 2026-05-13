import numpy as np
import torch
import open3d as o3d
from utils import farthest_point_sample

def compute_vertex_normals(vertices, faces):
    # Compute the face normals
    p0 = vertices[faces[:, 0], :]
    p1 = vertices[faces[:, 1], :]
    p2 = vertices[faces[:, 2], :]
    face_normals = torch.cross(p1 - p0, p2 - p0)
    face_normals = face_normals / torch.norm(face_normals, dim=1, keepdim=True)
    # Accumulate the normals for each vertex
    vertex_normals = torch.zeros_like(vertices)
    vertex_normals.index_add_(0, faces[:, 0], face_normals)
    vertex_normals.index_add_(0, faces[:, 1], face_normals)
    vertex_normals.index_add_(0, faces[:, 2], face_normals)
    # Normalize the accumulated normals
    vertex_normals = vertex_normals / torch.norm(vertex_normals, dim=1, keepdim=True)
    return vertex_normals

# mesh to partial
def full_to_partial(mesh,batch_size):
    target_mesh_F = mesh['faces'].squeeze()       # target_pcd.points = o3d.utility.Vector3dVector(data["shape2"]['vts'])
    target_mesh_V = mesh['vts'].squeeze()  # 5000 3

    # compute normals, keep forward-facing part
    target_normal = compute_vertex_normals(target_mesh_V,target_mesh_F)
    target_mesh_V = target_mesh_V[target_normal[:,2]>0] # 2514 3
    pcd_batched = np.expand_dims(arr, axis=0).repeat(batch_size, axis=0)
    return pcd_batched


# def fullpoint_to_partial(pcd_batch):
#     batch_size = pcd_batch.shape[0]
#     pcd_batched = np.zeros((batch_size, 1500, 3), dtype="float32")
#     for i in range(batch_size):
#         point = pcd_batch[i, :, :]
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(point.squeeze().cpu().numpy())

#         max_nn = 30
#         pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
#         new_points = np.asarray(pcd.points)[np.asarray(pcd.normals)[:,2]>0]
#         new_points= torch.tensor(new_points,dtype=torch.float32)
#         new_points_fps = farthest_point_sample(new_points, new_points.shape[0])
#         fps = new_points_fps[:1500].long()
#         new_points = new_points.unsqueeze(0)
#         point_sample = new_points[0, fps[0]]
#         pcd_batched[i] = point_sample
#     pcd_batched= torch.tensor(pcd_batched)
#     return pcd_batched

def fullpoint_to_partial(pcd_input):
    device = pcd_input.device
    pcd_np = o3d.geometry.PointCloud()
    pcd_np.points = o3d.utility.Vector3dVector(pcd_input.cpu().squeeze().numpy())
    radius = 0.5
    max_nn = 30
    pcd_np.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
    new_points = np.asarray(pcd_np.points)[np.asarray(pcd_np.normals)[:,2]>0]
    index = np.asarray(np.where((np.asarray(pcd_np.normals)[:,2]>0)>0))
    index = torch.from_numpy(index).to(device)
    arr = np.asarray(new_points)
    pcd_batched = torch.from_numpy(arr).to(device)
    return pcd_batched, index

def compute_vertex_normals(vertices, faces):
    """
    Computes the vertex normals of a mesh given its vertices and faces.
    vertices: a tensor of shape (num_vertices, 3) containing the 3D positions of the vertices
    faces: a tensor of shape (num_faces, 3) containing the vertex indices of each face
    returns: a tensor of shape (num_vertices, 3) containing the 3D normals of each vertex
    """
    # Compute the face normals
    p0 = vertices[faces[:, 0], :]
    p1 = vertices[faces[:, 1], :]
    p2 = vertices[faces[:, 2], :]
    face_normals = torch.cross(p1 - p0, p2 - p0)
    face_normals = face_normals / torch.norm(face_normals, dim=1, keepdim=True)
    # Accumulate the normals for each vertex
    vertex_normals = torch.zeros_like(vertices)
    vertex_normals.index_add_(0, faces[:, 0], face_normals)
    vertex_normals.index_add_(0, faces[:, 1], face_normals)
    vertex_normals.index_add_(0, faces[:, 2], face_normals)
    # Normalize the accumulated normals
    vertex_normals = vertex_normals / torch.norm(vertex_normals, dim=1, keepdim=True)
    return vertex_normals
