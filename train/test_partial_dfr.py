import argparse
import yaml
import os
import torch
from dataset_partial import DFRDataset_eval, shape_to_device
from utils import DQFMLoss, M2PLoss, DFRLoss_partial, farthest_point_sample, farthest_point_sample_batch
from sklearn.neighbors import NearestNeighbors
from models.DPFM import DGCNNFMNet
from models.dgcnn_sample import DecoderSimpleDGCNN, DecoderSimpleDGCNN_sample
from models.pointnet import PointNetBasis
from partial import fullpoint_to_partial
from AverageMeter import AverageMeter
from models.attention_net import CrossAttentionRefinementNet
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

import scipy


def convert_C_2(evals_x, evals_y, evecs_x, evecs_y, A1, A2, alpha, lambda_):
    # compute linear operator matrix representation C1 and C2
    evecs_x, evecs_y = evecs_x[:, :, :50], evecs_y[:, :, :50]
    evals_x, evals_y = evals_x[:, :50], evals_y[:, :50]

    T12 = knnsearch(A1, A2, alpha)
    T21 = knnsearch(A2, A1, alpha)

    # T12 = torch.bmm(torch.bmm(mask12, T12), mask21)
    # T21 = torch.bmm(torch.bmm(mask21, T21), mask12)

    evecs_x_t = evecs_x.transpose(1,2)
    evecs_y_t = evecs_y.transpose(1,2)

    evecs_y_t_y, evecs_x_t_x = torch.bmm(evecs_y_t, evecs_y), torch.bmm(evecs_x_t, evecs_x)

    evecs_y_t_T_x, evecs_x_t_T_y = torch.bmm(torch.bmm(evecs_y_t, T21), evecs_x), torch.bmm(torch.bmm(evecs_x_t, T12), evecs_y)

    C12_i = []
    for i in range(evals_x.size(1)):
        D12_i = torch.cat([torch.diag((evals_x[bs, i] - evals_y[bs]) * (evals_x[bs, i] - evals_y[bs])).unsqueeze(0) for bs in range(evals_x.size(0))], dim=0)
        C12 = torch.bmm(torch.inverse(evecs_y_t_y + lambda_ * D12_i), evecs_y_t_T_x[:, :, i].unsqueeze(2))
        C12_i.append(C12)
    C12 = torch.cat(C12_i, dim=2)
    C21_i = []
    for i in range(evals_y.size(1)):
        D21_i = torch.cat([torch.diag((evals_y[bs, i] - evals_x[bs]) * (evals_y[bs, i] - evals_x[bs])).unsqueeze(0) for bs in range(evals_y.size(0))], dim=0)
        C21 = torch.bmm(torch.inverse(evecs_x_t_x + lambda_ * D21_i), evecs_x_t_T_y[:, :, i].unsqueeze(2))
        C21_i.append(C21)
    C21 = torch.cat(C21_i, dim=2)
    return C12, C21

def euclidean_dist(x, y):
    bs, m, n = x.size(0), x.size(1), y.size(1)
    xx = torch.pow(x, 2).sum(2, keepdim=True).expand(bs, m, n)
    yy = torch.pow(y, 2).sum(2, keepdim=True).expand(bs, n, m).transpose(1, 2)
    dist = xx + yy - 2 * torch.bmm(x, y.transpose(1, 2))
    return dist

def knnsearch(x, y, alpha):
    distance = euclidean_dist(x, y)
    output = F.softmax(-alpha*distance, dim=-1)
    return output

def knnsearch_t(x, y):
    # distance = torch.cdist(x.float(), y.float())
    distance = torch.cdist(x.float(), y.float(), compute_mode='donot_use_mm_for_euclid_dist')
    _, idx = distance.topk(k=1, dim=-1, largest=False)
    return idx+1

def search_t(A1, A2):
    T12 = knnsearch_t(A1, A2)
    T21 = knnsearch_t(A2, A1)
    return T12

def convert_C(evals_x, evals_y, evecs_x, evecs_y, A1, A2, mask12, mask21, alpha, lambda_):
    # compute linear operator matrix representation C1 and C2
    evecs_x, evecs_y = evecs_x[:, :, :50], evecs_y[:, :, :50]
    evals_x, evals_y = evals_x[:, :50], evals_y[:, :50]

    T12 = knnsearch(A1, A2, alpha)
    T21 = knnsearch(A2, A1, alpha)

    # T12 = torch.bmm(torch.bmm(mask12, T12), mask21)
    # T21 = torch.bmm(torch.bmm(mask21, T21), mask12)

    evecs_x_t = evecs_x.transpose(1,2)
    evecs_y_t = evecs_y.transpose(1,2)

    evecs_y_t_y, evecs_x_t_x = torch.bmm(evecs_y_t, evecs_y), torch.bmm(evecs_x_t, evecs_x)

    evecs_y_t_T_x, evecs_x_t_T_y = torch.bmm(torch.bmm(evecs_y_t, T21), evecs_x), torch.bmm(torch.bmm(evecs_x_t, T12), evecs_y)

    C12_i = []
    for i in range(evals_x.size(1)):
        D12_i = torch.cat([torch.diag((evals_x[bs, i] - evals_y[bs]) * (evals_x[bs, i] - evals_y[bs])).unsqueeze(0) for bs in range(evals_x.size(0))], dim=0)
        C12 = torch.bmm(torch.inverse(evecs_y_t_y + lambda_ * D12_i), evecs_y_t_T_x[:, :, i].unsqueeze(2))
        C12_i.append(C12)
    C12 = torch.cat(C12_i, dim=2)
    C21_i = []
    for i in range(evals_y.size(1)):
        D21_i = torch.cat([torch.diag((evals_y[bs, i] - evals_x[bs]) * (evals_y[bs, i] - evals_x[bs])).unsqueeze(0) for bs in range(evals_y.size(0))], dim=0)
        C21 = torch.bmm(torch.inverse(evecs_x_t_x + lambda_ * D21_i), evecs_x_t_T_y[:, :, i].unsqueeze(2))
        C21_i.append(C21)
    C21 = torch.cat(C21_i, dim=2)
    return [C12, C21]

def z_aug(pc, pc_sample):
    # scale = np.diag(np.random.RandomState.uniform(1, 1, 3).astype(np.float32))
    rng = np.random.RandomState()
    angle = rng.uniform(-20, 20) / 180.0 * np.pi   ## multiway 
    rot_matrix = np.array([
        [np.cos(angle), 0., np.sin(angle)],
        [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]
    ], dtype=np.float32)
    # matrix = scale.dot(rot_matrix.T)
    device = pc.device
    bs = pc.shape[0]
    matrix = torch.from_numpy(rot_matrix).unsqueeze(0).repeat(bs, 1, 1).to(device)
    pc_rot = torch.bmm(pc, matrix)
    pc_sample_rot = torch.bmm(pc_sample, matrix)
    return pc_rot, pc_sample_rot

def eval_net(cfg):
    model_path = cfg["test"]["model_path"]
    save_path = os.path.join("results", cfg["expname"])
    if torch.cuda.is_available() and cfg["misc"]["cuda"]:
        device = torch.device(f'cuda:{cfg["misc"]["device"]}')
    else:
        device = torch.device("cpu")

    if cfg["templatename"] == "faust":
        usefaust = True
    else:
        usefaust = False
    # important paths
    base_path = os.path.dirname(__file__)
    op_cache_dir = os.path.join(base_path, cfg["dataset"]["cache_dir"])
    dataset_path_test = os.path.join(cfg["dataset"]["root_dataset"], cfg["target_data_set_name"])
    dataset_path_train = os.path.join(cfg["dataset"]["root_dataset"], cfg["source_data_set_name"])

    # decide on the use of WKS descriptors
    with_wks = None if cfg["fmap"]["C_in"] <= 3 else cfg["fmap"]["C_in"]

    # create dataset
    train_dataset = DFRDataset_eval(dataset_path_train, name=cfg["source_data_set_name"],
                                with_wks=with_wks, with_sym=cfg["dataset"]["with_sym"],
                                use_cache=True, op_cache_dir=op_cache_dir, class_name=cfg["dataset"]['class'], train=True,cfg=cfg,faustdata = usefaust)
    test_dataset = DFRDataset_eval(dataset_path_test, name=cfg["target_data_set_name"],
                                with_wks=with_wks, with_sym=cfg["dataset"]["with_sym"],
                                use_cache=True, op_cache_dir=op_cache_dir, class_name=cfg["dataset"]['class'], train=False,cfg=cfg,faustdata = False)

    
    # test loader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

    feat_refiner = CrossAttentionRefinementNet(n_in=cfg["fmap"]["n_feat"], num_head=cfg["attention"]["num_head"], gnn_dim=cfg["attention"]["gnn_dim"],
                                                overlap_feat_dim=cfg["overlap"]["overlap_feat_dim"],
                                                n_layers=cfg["attention"]["ref_n_layers"],
                                                cross_sampling_ratio=cfg["attention"]["cross_sampling_ratio"],
                                                attention_type=cfg["attention"]["attention_type"]).to(device)

    robust = cfg["fmap"]["robust"]
    if cfg["training"]['model'] == 'pointnet':
        point_backbone = PointNetBasis().to(device)
    elif cfg["training"]['model'] == 'dgcnn':
        point_backbone = DecoderSimpleDGCNN(device).to(device)
    elif cfg["training"]['model'] == 'dgcnnsample':
        point_backbone = DecoderSimpleDGCNN_sample(device=device).to(device)

    print(model_path)
    point_backbone = DGCNNFMNet(cfg).to(device) 
    #print(torch.load(model_path, map_location=device))
    point_backbone.load_state_dict(torch.load(model_path, map_location=device))
    point_backbone.eval()
    point_backbone = point_backbone.feature_extractor

    with torch.no_grad():
        if cfg["test"]["test_full2full"]:
            for i, data in tqdm(enumerate(test_loader)):
                data = shape_to_device(data, device)
                name1 = data["shape1"]["name"][0]
                name2 = data["shape2"]["name"][0]
                print(name1)
                print(name2)
                # full
                #idx1_partial = data["shape1"]['idx_partial'].unsqueeze(0) 
                V1 = data["shape1"]['verts']
                V2 = data["shape2"]['verts']
                V1_partial_sample, V2_sample = data["shape1"]['verts_sample'], data["shape2"]['verts_sample']
                V1_fps_sample, V2_fps_sample = data["shape1"]['fps_sample'], data["shape2"]['fps_sample']
                # sample
                feat1_p, feat2_p = point_backbone(V1.permute(0,2,1),V1_partial_sample.permute(0,2,1),V1_fps_sample), point_backbone(V2.permute(0,2,1),V2_sample.permute(0,2,1),V2_fps_sample)    
                # without partial
                #feat1_p, feat2_p = point_backbone(V1_partial.permute(0,2,1)), point_backbone(V2.permute(0,2,1))
                # refine feature
                # ref_feat1, ref_feat2 = feat_refiner(feat1_p, feat2_p)
                use_feat1, use_feat2 = feat1_p, feat2_p
                #lambda_ = 1e-1
                #alpha_i = 100
                T12_pred, T21_pred = search_t(use_feat1, use_feat2), search_t(use_feat2, use_feat1)

                # save files
                save_path_t = save_path + '/T/'
                if not os.path.exists(save_path_t):
                    os.makedirs(save_path_t)
                filename_t12 = f'T_{name1}_{name2}.txt'
                t12 = T12_pred.detach().cpu().squeeze(0).numpy()
                np.savetxt(os.path.join(save_path_t, filename_t12), t12, fmt='%i')
                filename_t21 = f'T_{name2}_{name1}.txt'
                t21 = T21_pred.detach().cpu().squeeze(0).numpy()
                np.savetxt(os.path.join(save_path_t, filename_t21), t21, fmt='%i')

                # full
                # save_path_p = save_path + '/Points_partial/'
                # if not os.path.exists(save_path_p):
                #     os.makedirs(save_path_p)
                # filename_p12 = f'{name1}_{name2}_partial_{name1}.npy'
                # P12 = V1_partial.detach().cpu().squeeze(0).numpy()
                # np.save(os.path.join(save_path_p, filename_p12), P12)

                # full
                # save_path_t = save_path + '/index_partial/'
                # if not os.path.exists(save_path_t):
                #     os.makedirs(save_path_t)
                # filename_index = f'{name1}_{name2}_index_{name1}.txt'
                # idx1p = idx1_partial.detach().cpu().squeeze(0).numpy()
                # np.savetxt(os.path.join(save_path_t, filename_index), idx1p, fmt='%i')

                save_path_t = save_path + '/feature/'
                if not os.path.exists(save_path_t):
                    os.makedirs(save_path_t)
                filename_index = f'{name1}_{name2}_usefeature_{name1}.mat'
                u_feat1 = use_feat1.detach().cpu().squeeze(0).numpy()
                u_feat1 = {'uphi': u_feat1}
                scipy.io.savemat(os.path.join(save_path_t, filename_index), u_feat1)
                filename_index = f'{name1}_{name2}_usefeature_{name2}.mat'
                u_feat2 = use_feat2.detach().cpu().squeeze(0).numpy()
                u_feat2 = {'uphi': u_feat2}
                scipy.io.savemat(os.path.join(save_path_t, filename_index), u_feat2)

        else:
            for i, data in tqdm(enumerate(train_loader)):
                # mesh000
                data = shape_to_device(data, device)
                name2 = data["shape2"]["name"][0]
                print(name2)
                V2 = data["shape2"]['verts'].to(device)
                V2_sample = data["shape2"]['verts_sample'].to(device)
                V2_fps_sample = data["shape2"]['fps_sample'] 
                # exit()
                break

            for i, data in tqdm(enumerate(test_loader)):
                data = shape_to_device(data, device)
                name1 = data["shape1"]["name"][0]
                print(name1)
                idx1_partial = data["shape1"]['idx_partial'].unsqueeze(0) 
                V1_partial = data["shape1"]['verts'].to(device)
                V1_partial_sample = data["shape1"]['verts_sample'].to(device)
                V1_fps_sample = data["shape1"]['fps_sample']
                print(idx1_partial.shape,V2_sample.shape)
                # sample
                feat1_p, feat2_p = point_backbone(V1_partial.permute(0,2,1),V1_partial_sample.permute(0,2,1),V1_fps_sample), point_backbone(V2.permute(0,2,1),V2_sample.permute(0,2,1),V2_fps_sample)    
                # without sample
                #feat1_p, feat2_p = point_backbone(V1_partial.permute(0,2,1)), point_backbone(V2.permute(0,2,1))
                # refine feature
                # ref_feat1, ref_feat2 = feat_refiner(feat1_p, feat2_p)
                use_feat1, use_feat2 = feat1_p, feat2_p
                #lambda_ = 1e-1
                #alpha_i = 100
                T12_pred, T21_pred = search_t(use_feat1, use_feat2), search_t(use_feat2, use_feat1)

                # save files
                save_path_t = save_path + '/T/'
                if not os.path.exists(save_path_t):
                    os.makedirs(save_path_t)
                filename_t12 = f'T_{name1}_{name2}.txt'
                t12 = T12_pred.detach().cpu().squeeze(0).numpy()
                np.savetxt(os.path.join(save_path_t, filename_t12), t12, fmt='%i')
                filename_t21 = f'T_{name2}_{name1}.txt'
                t21 = T21_pred.detach().cpu().squeeze(0).numpy()
                np.savetxt(os.path.join(save_path_t, filename_t21), t21, fmt='%i')

                save_path_p = save_path + '/Points_partial/'
                if not os.path.exists(save_path_p):
                    os.makedirs(save_path_p)
                filename_p12 = f'{name1}.npy'
                P12 = V1_partial.detach().cpu().squeeze(0).numpy()
                np.save(os.path.join(save_path_p, filename_p12), P12)

                save_path_t = save_path + '/index_partial/'
                if not os.path.exists(save_path_t):
                    os.makedirs(save_path_t)
                filename_index = f'index_{name1}.txt'
                idx1p = idx1_partial.detach().cpu().squeeze(0).numpy()
                np.savetxt(os.path.join(save_path_t, filename_index), idx1p, fmt='%i')

                save_path_t = save_path + '/feature/'
                if not os.path.exists(save_path_t):
                    os.makedirs(save_path_t)
                filename_index = f'usefeature_{name1}.mat'
                u_feat1 = use_feat1.detach().cpu().squeeze(0).numpy()
                u_feat1 = {'uphi': u_feat1}
                scipy.io.savemat(os.path.join(save_path_t, filename_index), u_feat1)
                filename_index = f'usefeature_{name2}.mat'
                u_feat2 = use_feat2.detach().cpu().squeeze(0).numpy()
                u_feat2 = {'uphi': u_feat2}
                scipy.io.savemat(os.path.join(save_path_t, filename_index), u_feat2)        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the testing of dgcnnsample model.")
    parser.add_argument('--savedir', required=False, default="./results", help='root directory of the dataset')
    parser.add_argument("--config", type=str, default="test", help="Config file name")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"./config/{args.config}.yaml", "r"))
    print(cfg)
    eval_net(cfg)