import argparse
import yaml
import os
import torch
from dataset_partial_fpcross import DFRDataset, shape_to_device
from utils import DQFMLoss, M2PLoss, LGLoss, farthest_point_sample, farthest_point_sample_batch,search_t
from sklearn.neighbors import NearestNeighbors
from models.dgcnn_sample import DecoderSimpleDGCNN, DecoderSimpleDGCNN_sample
from models.pointnet import PointNetBasis
from partial import fullpoint_to_partial
from AverageMeter import AverageMeter
from models.attention_net import CrossAttentionRefinementNet
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from lgattention import LGAttention_cross_new
from model import RegularizedFMNet

from tensorboardX import SummaryWriter
from collections import OrderedDict

def T2Pi(T12_init, T21_init,batch_size):
    device = T12_init.device
    n1, n2 = T12_init.shape[1], T21_init.shape[1]
    T12, T21 = torch.zeros(batch_size,n1, n2).to(device), torch.zeros(batch_size,n2, n1).to(device)
    L1= torch.arange(n1).to(device)
    L2 = torch.arange(n2).to(device)
    for i in range(batch_size):
        T12[i,L1,T12_init[i]] = 1
    for i in range(batch_size):
        T21[i,L2,T21_init[i]] = 1
    return T12, T21

def convert_C_full(Phi1, Phi2, A1, A2, alpha):
    Phi1, Phi2 = Phi1[:,:, :50], Phi2[:,:, :50]
    D1 = torch.bmm(Phi1, A1)
    D2 = torch.bmm(Phi2, A2)
    T12 = knnsearch(D1, D2, alpha)
    T21 = knnsearch(D2, D1, alpha)
    C12_new = torch.bmm(torch.pinverse(Phi2), torch.bmm(T21, Phi1))
    C21_new = torch.bmm(torch.pinverse(Phi1), torch.bmm(T12, Phi2))

    return C12_new, C21_new

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

def convert_C(evals_x, evals_y, evecs_x, evecs_y, A1, A2, alpha, lambda_):
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

def load_model(model, model_path, device):
    model_dict = torch.load(model_path, map_location=device)
    new_dict = OrderedDict()
    for k in model_dict.keys():
        key_new = k.replace('attention.','')
        # print(key_new)
        new_dict[key_new] = model_dict[k]
    model.load_state_dict(new_dict)    
    return model

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

    if cfg["training"]['model'] == 'pointnet':
        point_backbone = PointNetBasis().to(device)
    elif cfg["training"]['model'] == 'dgcnn':
        point_backbone = DecoderSimpleDGCNN(device).to(device)
    elif cfg["training"]['model'] == 'dgcnnsample':
        point_backbone = DecoderSimpleDGCNN_sample(device=device).to(device)
    elif cfg["training"]['model'] == 'lgattention':
        point_backbone = LGAttention_cross_new(kembed=40, k=40,emb_dims=512).to(device)
        if cfg.get("test", {}).get("model_path"):
            point_backbone = load_model(point_backbone, cfg["test"]["model_path"], device)

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

    lr = float(cfg["optimizer"]["lr"])
    optimizer = torch.optim.Adam(point_backbone.parameters(), lr=lr, betas=(cfg["optimizer"]["b1"], cfg["optimizer"]["b2"]))
    criterion = LGLoss(w_ortho=cfg["loss"]["w_ortho"],
                       w_bij=cfg["loss"]["w_bij"],
                       w_res=cfg["loss"]["w_res"],
                       w_gt=cfg["loss"]["w_gt"],
                       w_dist=cfg["loss"]["w_dist"]).to(device)
    
    fmreg_net = RegularizedFMNet(lambda_=cfg["fmap"]["lambda_"],
                                resolvant_gamma=cfg["fmap"]["resolvant_gamma"])

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
        train_loss, gt_loss_sum, dist_loss_sum, res_loss_sum, ortho_loss_sum, bij_loss_sum = 0, 0, 0, 0,0,0
        train_iters = 0
        alpha_i = alpha_list[epoch-1]

        lambda_ = 1e-1
        
        point_backbone.train()

        losses = AverageMeter(['Loss'])
        gt_losses = AverageMeter(['Loss'])
        ortho_losses = AverageMeter(['Loss'])
        res_losses = AverageMeter(['Loss'])
        dist_losses = AverageMeter(['Loss'])
        bij_losses = AverageMeter(['Loss'])
        val_losses = AverageMeter(['Loss'])

        for i, data in tqdm(enumerate(train_loader)):
            data = shape_to_device(data, device)
            V1_partial = data["shape1"]['verts']
            V1_full,V2_full = data["shape1"]['verts_full'],data["shape2"]['verts_full']
            evec_1,evec_2= data["shape1"]['phi_full'],data["shape2"]['phi_full']
            evecs1_partial= data["shape1"]['phi_partial']
            mass1,mass2 = data["shape1"]['mass'],data["shape2"]['mass']
            dist1,dist2 = data["shape1"]['dist'],data["shape2"]['dist']
            evals1,evals2 = data["shape1"]['eval'],data["shape2"]['eval']

            point_backbone.eval()
            feat1, feat2 = point_backbone(V1_full.permute(0,2,1)), point_backbone(V2_full.permute(0,2,1))
            T21,T12= search_t(feat1, feat2).squeeze(-1)-1, search_t(feat2, feat1).squeeze(-1)-1
            Pi12, Pi21 = T2Pi(T12,T21,V1_full.shape[0])
            C21_gt = torch.pinverse(evec_1[:,:, :50]) @ Pi21 @ evec_2[:,:, :50]
            point_backbone.train()

            evecs_trans1, evecs_trans2 = evec_1.transpose(-2, -1)[:,:cfg["fmap"]["n_fmap"]] @ torch.diag_embed(mass1), evec_2.transpose(-2, -1)[:,:cfg["fmap"]["n_fmap"]] @ torch.diag_embed(mass2)
            C12_pred, C21_pred = fmreg_net(feat1, feat2, evals1[:,:cfg["fmap"]["n_fmap"]], evals2[:,:cfg["fmap"]["n_fmap"]], evecs_trans1, evecs_trans2)
            A1 = torch.bmm(evecs_trans1, feat1)
            A2 = torch.bmm(evecs_trans2, feat2)
            C12_pred_new, C21_pred_new = convert_C_full(evec_1, evec_2, A1, A2, alpha_i)
            
            feat1_p, feat2_p = point_backbone(V1_partial.permute(0,2,1)), point_backbone(V2_full.permute(0,2,1))
            C12_p, C21_p = convert_C(evals1, evals2, evecs1_partial, evec_2, feat1_p, feat2_p, alpha_i, lambda_)
            loss,ortho_loss,bij_loss,res_loss,dist_loss,gt_loss = criterion(C12_pred.to(device), C21_pred.to(device), C12_pred_new.to(device), C21_pred_new.to(device),feat1,feat2,dist1,dist2,C21_gt, C21_p)
        
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_iters += 1
            losses.update([loss.item()])
            gt_losses.update([gt_loss.item()])
            ortho_losses.update([ortho_loss.item()])
            bij_losses.update([bij_loss.item()])
            res_losses.update([res_loss.item()])
            dist_losses.update([dist_loss.item()])
            
            train_loss += loss
            gt_loss_sum += gt_loss
            ortho_loss_sum +=  ortho_loss
            bij_loss_sum += bij_loss
            res_loss_sum +=  res_loss
            dist_loss_sum += dist_loss

            if train_iters % 20 == 0:
                print(f"iteration:{train_iters}, total_loss:{['%.4f' % l for l in losses.val()]}, gt_loss:{['%.4f' % l for l in gt_losses.val()]}, ortho_loss:{['%.4f' % l for l in ortho_losses.val()]}, bij_loss:{['%.4f' % l for l in bij_losses.val()]}, dist_loss:{['%.4f' % l for l in dist_losses.val()]}, res_loss:{['%.4f' % l for l in res_losses.val()]}")

        print(f"epoch:{epoch}, train_loss:{losses.avg(0)}, {bij_losses.avg(0)},dist_loss:{dist_losses.avg(0)}, res_loss:{res_losses.avg(0)}")
        
        if train_writer is not None:
            train_writer.add_scalar('train_Loss', (train_loss/train_iters).item(), epoch)
            train_writer.add_scalar('gt_loss', (gt_loss_sum/train_iters).item(), epoch)
            train_writer.add_scalar('ortho_oss', (ortho_loss_sum/train_iters).item(), epoch)
            train_writer.add_scalar('bij_loss', (bij_loss_sum/train_iters).item(), epoch)
            train_writer.add_scalar('dist_oss', (dist_loss_sum/train_iters).item(), epoch)
            train_writer.add_scalar('res_loss', (res_loss_sum/train_iters).item(), epoch)

        with torch.no_grad():
            point_backbone.eval()
            val_loss_sum = 0  
            val_iters = 0
            for i, data in tqdm(enumerate(test_loader)):
                data = shape_to_device(data, device)
                V1_partial = data["shape1"]['verts']
                V1_full,V2_full = data["shape1"]['verts_full'],data["shape2"]['verts_full']
                evec_1,evec_2= data["shape1"]['phi_full'],data["shape2"]['phi_full']
                mass1,mass2 = data["shape1"]['mass'],data["shape2"]['mass']
                dist1,dist2 = data["shape1"]['dist'],data["shape2"]['dist']
                evals1,evals2 = data["shape1"]['eval'],data["shape2"]['eval']
                evecs1_partial= data["shape1"]['phi_partial']


                feat1, feat2 = point_backbone(V1_full.permute(0,2,1)), point_backbone(V2_full.permute(0,2,1))
                T21,T12= search_t(feat1, feat2).squeeze(-1)-1, search_t(feat2, feat1).squeeze(-1)-1
                Pi12, Pi21 = T2Pi(T12,T21,V1_full.shape[0])
                C21_gt = torch.pinverse(evec_1[:,:, :50]) @ Pi21 @ evec_2[:,:, :50]

                evecs_trans1, evecs_trans2 = evec_1.transpose(-2, -1)[:,:cfg["fmap"]["n_fmap"]] @ torch.diag_embed(mass1), evec_2.transpose(-2, -1)[:,:cfg["fmap"]["n_fmap"]] @ torch.diag_embed(mass2)
                C12_pred, C21_pred = fmreg_net(feat1, feat2, evals1[:,:cfg["fmap"]["n_fmap"]], evals2[:,:cfg["fmap"]["n_fmap"]], evecs_trans1, evecs_trans2)
                A1 = torch.bmm(evecs_trans1, feat1)
                A2 = torch.bmm(evecs_trans2, feat2)
                C12_pred_new, C21_pred_new = convert_C_full(evec_1, evec_2, A1, A2, alpha_i)


                feat1_p, feat2_p = point_backbone(V1_partial.permute(0,2,1)), point_backbone(V2_full.permute(0,2,1))   
                C12_p, C21_p = convert_C(evals1, evals2, evecs1_partial, evec_2, feat1_p, feat2_p, alpha_i, lambda_)

                val_loss,ortho_loss,bij_loss,res_loss,dist_loss,gt_loss = criterion(C12_pred.to(device), C21_pred.to(device), C12_pred_new.to(device), C21_pred_new.to(device),feat1,feat2,dist1,dist2,C21_gt, C21_p)

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
    parser = argparse.ArgumentParser(description="Launch the training of lgattention model.")
    parser.add_argument('--savedir', required=False, default="./results", help='root directory of the dataset')
    parser.add_argument("--config", type=str, default="train", help="Config file name")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"./config/{args.config}.yaml", "r"))
    print(cfg)
    train_net(cfg)
