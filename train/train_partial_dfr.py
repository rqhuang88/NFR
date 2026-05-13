import argparse
import yaml
import os
import torch
from dataset_partial import DFRDataset, shape_to_device
from utils import DQFMLoss, M2PLoss, DFRLoss_partial, farthest_point_sample, farthest_point_sample_batch
from sklearn.neighbors import NearestNeighbors
from models.dgcnn_sample import DecoderSimpleDGCNN, DecoderSimpleDGCNN_sample
from models.pointnet import PointNetBasis
from partial import fullpoint_to_partial
from AverageMeter import AverageMeter
from models.attention_net import CrossAttentionRefinementNet
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from tensorboardX import SummaryWriter

def euclidean_dist(x, y):
    bs, m, n = x.size(0), x.size(1), y.size(1)
    xx = torch.pow(x, 2).sum(2, keepdim=True).expand(bs, m, n)
    yy = torch.pow(y, 2).sum(2, keepdim=True).expand(bs, n, m).transpose(1, 2)
    dist = xx + yy - 2 * torch.bmm(x, y.transpose(1, 2))
    return dist

def knnsearch(x, y, alpha):
    # distance = euclidean_dist(x, y)
    distance = torch.cdist(x,y)
    output = F.softmax(-alpha*distance, dim=-1)
    return output

def convert_C(evals_x, evals_y, evecs_x, evecs_y, A1, A2, mask12, mask21, alpha, lambda_):
    # compute linear operator matrix representation C1 and C2
    evecs_x, evecs_y = evecs_x[:, :, :50], evecs_y[:, :, :50]
    evals_x, evals_y = evals_x[:, :50], evals_y[:, :50]

    T12 = knnsearch(A1, A2, alpha)
    T21 = knnsearch(A2, A1, alpha)

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

def convert_C_2(evals_x, evals_y, evecs_x, evecs_y, A1, A2, alpha, lambda_):
    evecs_x, evecs_y = evecs_x[:, :, :50], evecs_y[:, :, :50]
    evals_x, evals_y = evals_x[:, :50], evals_y[:, :50]

    T12 = knnsearch(A1, A2, alpha)
    # T21 = knnsearch(A2, A1, alpha)

    evecs_x_t = evecs_x.transpose(1,2)
    # evecs_y_t = evecs_y.transpose(1,2)

    evecs_x_t_x = torch.bmm(evecs_x_t, evecs_x)

    evecs_x_t_T_y = torch.bmm(torch.bmm(evecs_x_t, T12), evecs_y)

    # C12_i = []
    # for i in range(evals_x.size(1)):
    #     D12_i = torch.cat([torch.diag((evals_x[bs, i] - evals_y[bs]) * (evals_x[bs, i] - evals_y[bs])).unsqueeze(0) for bs in range(evals_x.size(0))], dim=0)
    #     C12 = torch.bmm(torch.inverse(evecs_y_t_y + lambda_ * D12_i), evecs_y_t_T_x[:, :, i].unsqueeze(2))
    #     C12_i.append(C12)
    # C12 = torch.cat(C12_i, dim=2)
    C21_i = []
    for i in range(evals_y.size(1)):
        D21_i = torch.cat([torch.diag((evals_y[bs, i] - evals_x[bs]) * (evals_y[bs, i] - evals_x[bs])).unsqueeze(0) for bs in range(evals_y.size(0))], dim=0)
        C21 = torch.bmm(torch.inverse(evecs_x_t_x + lambda_ * D21_i), evecs_x_t_T_y[:, :, i].unsqueeze(2))
        C21_i.append(C21)
    C21 = torch.cat(C21_i, dim=2)
    return None, C21

def z_aug(pc, pc_sample):
    rng = np.random.RandomState()
    angle = rng.uniform(-20, 20) / 180.0 * np.pi   ## multiway 
    rot_matrix = np.array([
        [np.cos(angle), 0., np.sin(angle)],
        [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]
    ], dtype=np.float32)
    device = pc.device
    bs = pc.shape[0]
    matrix = torch.from_numpy(rot_matrix).unsqueeze(0).repeat(bs, 1, 1).to(device)
    pc_rot = torch.bmm(pc, matrix)
    pc_sample_rot = torch.bmm(pc_sample, matrix)
    return pc_rot, pc_sample_rot

def train_net(cfg):
    if torch.cuda.is_available() and cfg["misc"]["cuda"]:
        device = torch.device(f'cuda:{cfg["misc"]["device"]}')
    else:
        device = torch.device("cpu")

    # important paths
    base_path = os.path.dirname(__file__)
    op_cache_dir = os.path.join(base_path, cfg["dataset"]["cache_dir"])
    dataset_path_train = os.path.join(cfg["dataset"]["root_dataset"], cfg["dataset"]["root_train"])
    dataset_path_test = os.path.join(cfg["dataset"]["root_dataset"], cfg["dataset"]["root_test"])

    save_dir_name = cfg["expname"]
    model_save_path = os.path.join(base_path, f"ckpt/{save_dir_name}/ep" + "_{}.pth")
    if not os.path.exists(os.path.join(base_path, f"ckpt/{save_dir_name}/")):
        os.makedirs(os.path.join(base_path, f"ckpt/{save_dir_name}/"))

    # decide on the use of WKS descriptors
    with_wks = None if cfg["fmap"]["C_in"] <= 3 else cfg["fmap"]["C_in"]

    train_writer = SummaryWriter(os.path.join(base_path, 'tensorboard', save_dir_name))

    # create dataset
    train_dataset = DFRDataset(dataset_path_train, name=cfg["dataset"]["name"],
                                 with_wks=with_wks, with_sym=cfg["dataset"]["with_sym"],
                                 use_cache=True, op_cache_dir=op_cache_dir, class_name=cfg["dataset"]['class'], train=True,cfg=cfg)

    test_dataset = DFRDataset(dataset_path_test, name=cfg["dataset"]["name"],
                                with_wks=with_wks, with_sym=cfg["dataset"]["with_sym"],
                                use_cache=True, op_cache_dir=op_cache_dir, class_name=cfg["dataset"]['class'], train=False,cfg=cfg)


    # data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg["training"]['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg["training"]['batch_size'], shuffle=False)

    
    if cfg["training"]['model'] == 'pointnet':
        point_backbone = PointNetBasis().to(device)
    elif cfg["training"]['model'] == 'dgcnn':
        point_backbone = DecoderSimpleDGCNN(device).to(device)
    elif cfg["training"]['model'] == 'dgcnnsample':
        point_backbone = DecoderSimpleDGCNN_sample(device=device).to(device)
        
    lr = float(cfg["optimizer"]["lr"])
    optimizer = torch.optim.Adam(point_backbone.parameters(), lr=lr, betas=(cfg["optimizer"]["b1"], cfg["optimizer"]["b2"]))
    criterion = DFRLoss_partial(w_gt=cfg["loss"]["w_gt"],
                                w_ortho=cfg["loss"]["w_ortho"],
                                w_nce=cfg["loss"]["w_nce"],
                                w_bij=cfg["loss"]["w_bij"]).to(device)

    feat_refiner = CrossAttentionRefinementNet(n_in=cfg["fmap"]["n_feat"], num_head=cfg["attention"]["num_head"], gnn_dim=cfg["attention"]["gnn_dim"],
                                                        overlap_feat_dim=cfg["overlap"]["overlap_feat_dim"],
                                                        n_layers=cfg["attention"]["ref_n_layers"],
                                                        cross_sampling_ratio=cfg["attention"]["cross_sampling_ratio"],
                                                        attention_type=cfg["attention"]["attention_type"]).to(device)

    robust = cfg["fmap"]["robust"]

    # Training loop
    print("start training")
    alpha_list = np.linspace(cfg["loss"]["min_alpha"], cfg["loss"]["max_alpha"]+1, cfg["training"]["epochs"])
    val_best_loss = 1e10
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        if epoch % cfg["optimizer"]["decay_iter"] == 0:
            lr *= cfg["optimizer"]["decay_factor"]
            print(f"Decaying learning rate, new one: {lr}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        train_loss, gt_loss_sum, ortho_loss_sum, bij_loss_sum = 0, 0, 0, 0
        train_iters = 0
        alpha_i = alpha_list[epoch-1]

        lambda_ = 1e-1
        
        point_backbone.train()

        losses = AverageMeter(['Loss'])
        gt_losses = AverageMeter(['Loss'])
        ortho_losses = AverageMeter(['Loss'])
        bij_losses = AverageMeter(['Loss'])
        val_losses = AverageMeter(['Loss'])

        for i, data in tqdm(enumerate(train_loader)):
            data = shape_to_device(data, device)
            feat1_m, feat2_m = None, None
            C12_gt, C21_gt = data["C12_gt"], data["C21_gt"]
            mask12, mask21 = data["mask12"], data["mask21"]
            V1_eval, V2_eval  = data['shape1']['eval'], data['shape2']['eval']

            V1_partial, V2 = data["shape1"]['verts'], data["shape2"]['verts']
            V1_partial_sample, V2_sample = data["shape1"]['verts_sample'], data["shape2"]['verts_sample']
            V1_fps_sample, V2_fps_sample = data["shape1"]['fps_sample'], data["shape2"]['fps_sample']
            # V1_partial = data["shape1"]['verts_partial']
            # print(V1_partial_sample.device,V2_sample.device,V1_fps_sample.device,V2_fps_sample.device)
            evecs1_partial= data["shape1"]['phi_partial']
            evecs2 =  data["shape2"]['phi']

            print(data["shape1"]["name"])
            print(data["shape2"]["name"])

            # dgcnn feature
            # V1partial->V2
            # feat1_p, feat2_p = point_backbone(V1_partial.permute(0,2,1)), point_backbone(V2.permute(0,2,1))
            feat1_p, feat2_p = point_backbone(V1_partial.permute(0,2,1),V1_partial_sample.permute(0,2,1),V1_fps_sample), point_backbone(V2.permute(0,2,1),V2_sample.permute(0,2,1),V2_fps_sample)
            # refine feature
            # ref_feat1, ref_feat2 = feat_refiner(feat1_p, feat2_p)
            # use_feat1, use_feat2 = (ref_feat1, ref_feat2) if robust else (feat1_p, feat2_p)
            use_feat1, use_feat2 = feat1_p, feat2_p
            # cal C with reg
            # V1partial->V2  
            C12_p, C21_p = convert_C(V1_eval, V2_eval, evecs1_partial, evecs2, use_feat1, use_feat2, mask12, mask21, alpha_i, lambda_)

            loss, gt_loss, ortho_loss, bij_loss = criterion(C21_gt, C21_p, use_feat1, use_feat2, feat1_p, feat2_p)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_iters += 1

            losses.update([loss.item()])
            gt_losses.update([gt_loss.item()])
            ortho_losses.update([ortho_loss.item()])
            bij_losses.update([bij_loss.item()])
            
            train_loss += loss
            gt_loss_sum += gt_loss
            ortho_loss_sum +=  ortho_loss
            bij_loss_sum += bij_loss

            if train_iters % 20 == 0:
                print(f"iteration:{train_iters}, total_loss:{['%.4f' % l for l in losses.val()]}, gt_loss:{['%.4f' % l for l in gt_losses.val()]}, ortho_loss:{['%.4f' % l for l in ortho_losses.val()]}, bij_loss:{['%.4f' % l for l in bij_losses.val()]}")

        print(f"epoch:{epoch}, train_loss:{losses.avg(0)}, gt_loss:{gt_losses.avg(0)}, ortho_loss:{ortho_losses.avg(0)}, bij_loss:{bij_losses.avg(0)}")
        
        if train_writer is not None:
            train_writer.add_scalar('train_Loss', (train_loss/train_iters).item(), epoch)
            train_writer.add_scalar('gt_loss', (gt_loss_sum/train_iters).item(), epoch)
            train_writer.add_scalar('ortho_oss', (ortho_loss_sum/train_iters).item(), epoch)
            train_writer.add_scalar('bij_loss', (bij_loss_sum/train_iters).item(), epoch)

        with torch.no_grad():
            point_backbone.eval()
            val_loss_sum = 0  
            val_iters = 0
            for i, data in tqdm(enumerate(test_loader)):
                data = shape_to_device(data, device)

                # V1, V2 = data["shape1"]['verts'], data["shape2"]['verts']
                feat1_m, feat2_m = None, None
                # evecs1, evecs2 = data["shape1"]['phi'], data["shape2"]['phi']
                C12_gt, C21_gt = data["C12_gt"], data["C21_gt"]
                mask12, mask21 = data["mask12"], data["mask21"]
                V1_eval, V2_eval  = data['shape1']['eval'], data['shape2']['eval']
                V1_partial, V2 = data["shape1"]['verts'], data["shape2"]['verts']
                # V1_partial = data["shape1"]['verts_partial']
                V1_partial_sample, V2_sample = data["shape1"]['verts_sample'], data["shape2"]['verts_sample']
                V1_fps_sample, V2_fps_sample = data["shape1"]['fps_sample'], data["shape2"]['fps_sample']
                evecs1_partial= data["shape1"]['phi_partial']
                evecs2 =  data["shape2"]['phi']
                print(data["shape1"]["name"])
                print(data["shape2"]["name"])
                # change to point2s
    
                # V1partial->V2               
                # feat1_p, feat2_p = point_backbone(V1_partial.permute(0,2,1)), point_backbone(V2.permute(0,2,1))
                
                feat1_p, feat2_p = point_backbone(V1_partial.permute(0,2,1),V1_partial_sample.permute(0,2,1),V1_fps_sample), point_backbone(V2.permute(0,2,1),V2_sample.permute(0,2,1),V2_fps_sample)    
                # refine feature
                # ref_feat1, ref_feat2 = feat_refiner(feat1_p, feat2_p)
                # use_feat1, use_feat2 = (ref_feat1, ref_feat2) if robust else (feat1_p, feat2_p)
                use_feat1, use_feat2 = feat1_p, feat2_p
                # V1partial->V2
                C12_p, C21_p = convert_C(V1_eval, V2_eval, evecs1_partial, evecs2, use_feat1, use_feat2, mask12, mask21, alpha_i, lambda_)

    
                val_loss, gt_loss, ortho_loss, nce_loss = criterion(C21_gt, C21_p, use_feat1, use_feat2, feat1_p, feat2_p)

                val_iters += 1
                val_loss_sum += val_loss
                val_losses.update([val_loss.item()])
                
            
            print(f"epoch:{epoch}, val_loss:{val_loss.item()}")
            if train_writer is not None:
                train_writer.add_scalar('val_Loss', (val_loss_sum/val_iters).item(), epoch)
            

        # save model
        if (epoch + 1) % cfg["misc"]["checkpoint_interval"] == 0:
            torch.save(point_backbone.state_dict(), model_save_path.format(epoch))
        if val_loss_sum <= val_best_loss:
            val_best_loss = val_loss_sum
            torch.save(point_backbone.state_dict(), model_save_path.format('val_best'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the training of dgcnnsample model.")
    parser.add_argument('--savedir', required=False, default="./results", help='root directory of the dataset')
    parser.add_argument("--config", type=str, default="train", help="Config file name")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"./config/{args.config}.yaml", "r"))
    print(cfg)
    train_net(cfg)
