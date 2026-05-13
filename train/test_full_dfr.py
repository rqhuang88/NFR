import argparse
import yaml
import os
import torch
from dataset_partial_fpcross import DFRDataset_eval, shape_to_device
from utils import DQFMLoss, M2PLoss, farthest_point_sample, farthest_point_sample_batch
from sklearn.neighbors import NearestNeighbors
from models.DPFM import DGCNNFMNet
from models.dgcnn_sample import DecoderSimpleDGCNN, DecoderSimpleDGCNN_sample
from models.pointnet import PointNetBasis
from AverageMeter import AverageMeter
from models.attention_net import CrossAttentionRefinementNet
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import scipy


def knnsearch_t(x, y):
    distance = torch.cdist(x.float(), y.float(), compute_mode='donot_use_mm_for_euclid_dist')
    _, idx = distance.topk(k=1, dim=-1, largest=False)
    return idx + 1


def search_t(A1, A2):
    T12 = knnsearch_t(A1, A2)
    return T12


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

    base_path = os.path.dirname(__file__)
    op_cache_dir = os.path.join(base_path, cfg["dataset"]["cache_dir"])
    dataset_path_test = os.path.join(cfg["dataset"]["root_dataset"], cfg["target_data_set_name"])
    dataset_path_train = os.path.join(cfg["dataset"]["root_dataset"], cfg["source_data_set_name"])

    with_wks = None if cfg["fmap"]["C_in"] <= 3 else cfg["fmap"]["C_in"]

    train_dataset = DFRDataset_eval(dataset_path_train, name=cfg["source_data_set_name"],
                                with_wks=with_wks, with_sym=cfg["dataset"]["with_sym"],
                                use_cache=True, op_cache_dir=op_cache_dir, class_name=cfg["dataset"]['class'], train=True, cfg=cfg, faustdata=usefaust)
    test_dataset = DFRDataset_eval(dataset_path_test, name=cfg["target_data_set_name"],
                                with_wks=with_wks, with_sym=cfg["dataset"]["with_sym"],
                                use_cache=True, op_cache_dir=op_cache_dir, class_name=cfg["dataset"]['class'], train=False, cfg=cfg, faustdata=False)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

    feat_refiner = CrossAttentionRefinementNet(n_in=cfg["fmap"]["n_feat"], num_head=cfg["attention"]["num_head"], gnn_dim=cfg["attention"]["gnn_dim"],
                                                overlap_feat_dim=cfg["overlap"]["overlap_feat_dim"],
                                                n_layers=cfg["attention"]["ref_n_layers"],
                                                cross_sampling_ratio=cfg["attention"]["cross_sampling_ratio"],
                                                attention_type=cfg["attention"]["attention_type"]).to(device)

    if cfg["training"]['model'] == 'pointnet':
        point_backbone = PointNetBasis().to(device)
    elif cfg["training"]['model'] == 'dgcnn':
        point_backbone = DecoderSimpleDGCNN(device).to(device)
    elif cfg["training"]['model'] == 'dgcnnsample':
        point_backbone = DecoderSimpleDGCNN_sample(device=device).to(device)

    print(model_path)
    point_backbone = DGCNNFMNet(cfg).to(device)
    point_backbone.load_state_dict(torch.load(model_path, map_location=device))
    point_backbone.eval()
    point_backbone = point_backbone.feature_extractor

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            data = shape_to_device(data, device)
            name1 = data["shape1"]["name"][0]
            name2 = data["shape2"]["name"][0]
            print(name1, name2)

            V1 = data["shape1"]['verts']
            V2 = data["shape2"]['verts']
            V1_sample, V2_sample = data["shape1"]['verts_sample'], data["shape2"]['verts_sample']
            V1_fps_sample, V2_fps_sample = data["shape1"]['fps_sample'], data["shape2"]['fps_sample']

            feat1_p = point_backbone(V1.permute(0, 2, 1), V1_sample.permute(0, 2, 1), V1_fps_sample)
            feat2_p = point_backbone(V2.permute(0, 2, 1), V2_sample.permute(0, 2, 1), V2_fps_sample)
            use_feat1, use_feat2 = feat1_p, feat2_p
            T12_pred = search_t(use_feat1, use_feat2)
            T21_pred = search_t(use_feat2, use_feat1)

            save_path_t = os.path.join(save_path, 'T')
            if not os.path.exists(save_path_t):
                os.makedirs(save_path_t)
            filename_t12 = f'T_{name1}_{name2}.txt'
            t12 = T12_pred.detach().cpu().squeeze(0).numpy()
            np.savetxt(os.path.join(save_path_t, filename_t12), t12, fmt='%i')
            filename_t21 = f'T_{name2}_{name1}.txt'
            t21 = T21_pred.detach().cpu().squeeze(0).numpy()
            np.savetxt(os.path.join(save_path_t, filename_t21), t21, fmt='%i')

            save_path_f = os.path.join(save_path, 'feature')
            if not os.path.exists(save_path_f):
                os.makedirs(save_path_f)
            filename_f1 = f'{name1}_{name2}_usefeature_{name1}.mat'
            u_feat1 = use_feat1.detach().cpu().squeeze(0).numpy()
            scipy.io.savemat(os.path.join(save_path_f, filename_f1), {'uphi': u_feat1})
            filename_f2 = f'{name1}_{name2}_usefeature_{name2}.mat'
            u_feat2 = use_feat2.detach().cpu().squeeze(0).numpy()
            scipy.io.savemat(os.path.join(save_path_f, filename_f2), {'uphi': u_feat2})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the full DFR feature extractor.")
    parser.add_argument("--config", type=str, default="train_full_sf", help="Config file name")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"./config/{args.config}.yaml", "r"))
    print(cfg)
    eval_net(cfg)
