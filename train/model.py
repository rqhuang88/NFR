from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from lgattention import LGAttention,LGAttention_sample_new,LGAttention_sample_depth,LGAttention_cross_new
from einops import rearrange, repeat
# feature extractor
from diffusion_net.layers import DiffusionNet
# maps block
from utils import get_mask, nn_interpolate


class RegularizedFMNet(nn.Module):
    """Compute the functional map matrix representation."""

    def __init__(self, lambda_=1e-3, resolvant_gamma=0.5):
        super().__init__()
        self.lambda_ = lambda_
        self.resolvant_gamma = resolvant_gamma

    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        # compute linear operator matrix representation C1 and C2
        evecs_trans_x, evecs_trans_y = evecs_trans_x, evecs_trans_y
        evals_x, evals_y = evals_x, evals_y

        F_hat = torch.bmm(evecs_trans_x, feat_x)
        G_hat = torch.bmm(evecs_trans_y, feat_y)
        A, B = F_hat, G_hat
        
        D12 = torch.ones((evals_x.size(0), evals_x.size(1), evals_x.size(1)), device='cuda:0')
        D21 = torch.ones((evals_y.size(0), evals_y.size(1), evals_y.size(1)), device='cuda:0')
        for bs in range(evals_x.size(0)):
            D12[bs,:,:] = get_mask(evals_x[bs,:].flatten(), evals_y[bs,:].flatten(), self.resolvant_gamma, feat_x.device)
        for bs in range(evals_y.size(0)):
            D21[bs,:,:] = get_mask(evals_y[bs,:].flatten(), evals_x[bs,:].flatten(), self.resolvant_gamma, feat_x.device)

        A_t, B_t = A.transpose(1, 2), B.transpose(1, 2)
        A_A_t, B_B_t = torch.bmm(A, A_t), torch.bmm(B, B_t)
        B_A_t, A_B_t = torch.bmm(B, A_t), torch.bmm(A, B_t)

        # print(D12.shape)
        # print(evals_x.shape)
        # print(B_A_t.shape)

        C12_i = []
        for i in range(evals_x.size(1)):
            D12_i = torch.cat([torch.diag(D12[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_x.size(0))], dim=0)
            # print(D12_i.shape)
            C12 = torch.bmm(torch.inverse(A_A_t + self.lambda_ * D12_i), B_A_t[:, i, :].unsqueeze(1).transpose(1, 2))
            C12_i.append(C12.transpose(1, 2))
        C12 = torch.cat(C12_i, dim=1)
        # print(C12.shape)
        C21_i = []
        for i in range(evals_y.size(1)):
            D21_i = torch.cat([torch.diag(D21[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_y.size(0))], dim=0)
            C21 = torch.bmm(torch.inverse(B_B_t + self.lambda_ * D21_i), A_B_t[:, i, :].unsqueeze(1).transpose(1, 2))
            C21_i.append(C21.transpose(1, 2))
        C21 = torch.cat(C21_i, dim=1)

        return [C12, C21]

class RegularizedCFMNet(nn.Module):
    """Compute the complex functional map matrix representation."""

    def __init__(self, lambda_=1e-3, resolvant_gamma=0.5):
        super().__init__()
        self.lambda_ = lambda_
        self.resolvant_gamma = resolvant_gamma

    def forward(self, feat_x, feat_y, spec_grad_x, spec_grad_y, cevals_x, cevals_y):
        # compute linear operator matrix representation C1 and C2
        cty = torch.complex128
        spec_grad_x, spec_grad_y = spec_grad_x, spec_grad_y

        F_hat = torch.bmm(spec_grad_x, feat_x.type(cty))
        G_hat = torch.bmm(spec_grad_y, feat_y.type(cty))
        A, B = F_hat, G_hat

        # if normalize input vector fields
        # A, B = A/torch.abs(A), B/torch.abs(B)

        if self.lambda_ == 0:
            Q = (B @ torch.pinverse(A))
            return Q

        # else
        cevals_x, cevals_y = cevals_x, cevals_y
        D = get_mask(cevals_x.flatten(), cevals_y.flatten(), self.resolvant_gamma, feat_x.device)

        A_t = torch.conj(A.transpose(1, 2))
        A_A_t = torch.bmm(A, A_t)
        B_A_t = torch.bmm(B, A_t)

        Q_i = []
        for i in range(cevals_x.size(1)):
            D_i = torch.cat([torch.diag(D[bs, i, :].flatten()) for bs in range(cevals_x.size(0))], dim=0)
            Q = torch.bmm(torch.inverse(A_A_t + self.lambda_ * D_i),
                          torch.conj(B_A_t[:, i, :].transpose(1, 2)))
            Q_i.append(torch.conj(Q.transpose(1, 2)))
        Q = torch.cat(Q_i, dim=1)

        return Q

device = torch.device(f'cuda:0')
class DQFMNet(nn.Module):
    """
    Compilation of the global model :
    - diffusion net as feature extractor
    - fmap + q-fmap
    - unsupervised loss
    """
    # device = 'cuda:0'
    def __init__(self, cfg):
        super().__init__()
        # feature extractor #
        # with_grad=True
        #self.attention = LGAttention_sample(kembed=40,k=40,emb_dims=512)
        # 40,40
        self.attention = LGAttention_cross_new(kembed=20,k=20,emb_dims=512)

        # regularized fmap
        self.fmreg_net = RegularizedFMNet(lambda_=cfg["fmap"]["lambda_"],
                                          resolvant_gamma=cfg["fmap"]["resolvant_gamma"])
        # self.cfmreg_net = RegularizedCFMNet(lambda_=cfg["fmap"]["lambda_"],
        #                                     resolvant_gamma=cfg["fmap"]["resolvant_gamma"])
        # parameters
        self.n_fmap = cfg["fmap"]["n_fmap"]
        # self.n_cfmap = cfg["fmap"]["n_cfmap"]
        # self.robust = cfg["fmap"]["robust"]
        
    def forward(self, batch):
        verts1, mass1,evals1, evecs1= (batch["shape1"]["xyz"],
                                        batch["shape1"]["mass"],
                                        batch["shape1"]["evals"], batch["shape1"]["evecs"])
        verts2, mass2,evals2, evecs2= (batch["shape2"]["xyz"],
                                        batch["shape2"]["mass"],
                                        batch["shape2"]["evals"], batch["shape2"]["evecs"])

        verts1 = verts1.permute(0,2,1)
        verts2 = verts2.permute(0,2,1)

        feat1 = self.attention(verts1)
        feat2 = self.attention(verts2)

        evecs_trans1, evecs_trans2 = evecs1.transpose(-2, -1)[:,:self.n_fmap] @ torch.diag_embed(mass1), evecs2.transpose(-2, -1)[:,:self.n_fmap] @ torch.diag_embed(mass2)
        evals1, evals2 = evals1[:,:self.n_fmap], evals2[:,:self.n_fmap]

        #
        C12_pred, C21_pred = self.fmreg_net(feat1, feat2, evals1, evals2, evecs_trans1, evecs_trans2)
        #

        # if we don't have complex spectral info we just return C
        # if self.n_cfmap == 0:
        return C12_pred, C21_pred, None, feat1, feat2, evecs_trans1, evecs_trans2, evecs1, evecs2

        # # else, also predict cfmap
        # spec_grad1, spec_grad2 = batch["shape1"]["spec_grad"][:self.n_cfmap], batch["shape2"]["spec_grad"][:self.n_cfmap]
        # cevals1, cevals2 = batch["shape1"]["cevals"][:self.n_fmap], batch["shape2"]["cevals"][:self.n_fmap]
        # #

        # cfeat1, cfeat2 = feat1, feat2  # network features
        # Q_pred = self.cfmreg_net(cfeat1, cfeat2, spec_grad1, spec_grad2, cevals1, cevals2)

        # return C12_pred, C21_pred, Q_pred, feat1, feat2, evecs_trans1, evecs_trans2
