from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

# feature extractor

# maps block
from utils import get_mask, nn_interpolate


class RegularizedFMNet(nn.Module):
    """Compute the functional map matrix representation in DPFM"""
    def __init__(self, lambda_=100, resolvant_gamma=0.5, bidirectional=True):
        super(RegularizedFMNet, self).__init__()
        self.lmbda = lambda_
        self.resolvant_gamma = resolvant_gamma
        self.bidirectional = bidirectional

    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        """
        Forward pass to compute functional map
        Args:
            feat_x (torch.Tensor): feature vector of shape x. [B, Vx, C].
            feat_y (torch.Tensor): feature vector of shape y. [B, Vy, C].
            evals_x (torch.Tensor): eigenvalues of shape x. [B, K].
            evals_y (torch.Tensor): eigenvalues of shape y. [B, K].
            evecs_trans_x (torch.Tensor): pseudo inverse of eigenvectors of shape x. [B, K, Vx].
            evecs_trans_y (torch.Tensor): pseudo inverse of eigenvectors of shape y. [B, K, Vy].

        Returns:
            C (torch.Tensor): functional map from shape x to shape y. [B, K, K].
        """
        # print(feat_x.shape, feat_y.shape, evals_x.shape, evals_y.shape, evecs_trans_x.shape, evecs_trans_y.shape)
        
        A = torch.bmm(evecs_trans_x, feat_x)  # [B, K, C]
        B = torch.bmm(evecs_trans_y, feat_y)  # [B, K, C]

        D = get_mask(evals_x, evals_y, self.resolvant_gamma)  # [B, K, K]
        # print(D.shape)
        # exit()
        A_t = A.transpose(1, 2)  # [B, C, K]
        A_A_t = torch.bmm(A, A_t)  # [B, K, K]
        B_A_t = torch.bmm(B, A_t)  # [B, K, K]

        C_i = []
        for i in range(evals_x.shape[1]):
            D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_x.shape[0])], dim=0)
            C = torch.bmm(torch.inverse(A_A_t + self.lmbda * D_i), B_A_t[:, [i], :].transpose(1, 2))
            C_i.append(C.transpose(1, 2))

        Cxy = torch.cat(C_i, dim=1)

        if self.bidirectional:
            D = get_mask(evals_y, evals_x, self.resolvant_gamma)  # [B, K, K]

            B_t = B.transpose(1, 2)  # [B, C, K]
            B_B_t = torch.bmm(B, B_t)  # [B, K, K]
            A_B_t = torch.bmm(A, B_t)  # [B, K, K]

            C_i = []
            for i in range(evals_y.shape[1]):
                D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_y.shape[0])],
                                dim=0)
                C = torch.bmm(torch.inverse(B_B_t + self.lmbda * D_i), A_B_t[:, [i], :].transpose(1, 2))
                C_i.append(C.transpose(1, 2))

            Cyx = torch.cat(C_i, dim=1)
        else:
            Cyx = None

        return Cxy, Cyx







# class RegularizedFMNet(nn.Module):
#     """Compute the functional map matrix representation."""

#     def __init__(self, lambda_=1e-3, resolvant_gamma=0.5):
#         super().__init__()
#         self.lambda_ = lambda_
#         self.resolvant_gamma = resolvant_gamma

#     def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
#         # compute linear operator matrix representation C1 and C2
#         # evecs_trans_x, evecs_trans_y = evecs_trans_x.unsqueeze(0), evecs_trans_y.unsqueeze(0)
#         # evals_x, evals_y = evals_x.unsqueeze(0), evals_y.unsqueeze(0)

#         F_hat = torch.bmm(evecs_trans_x, feat_x)
#         G_hat = torch.bmm(evecs_trans_y, feat_y)
#         A, B = F_hat, G_hat

#         D12 = get_mask(evals_x.flatten(), evals_y.flatten(), self.resolvant_gamma, feat_x.device)
#         D21 = get_mask(evals_y.flatten(), evals_x.flatten(), self.resolvant_gamma, feat_x.device)

#         A_t, B_t = A.transpose(1, 2), B.transpose(1, 2)
#         A_A_t, B_B_t = torch.bmm(A, A_t), torch.bmm(B, B_t)
#         B_A_t, A_B_t = torch.bmm(B, A_t), torch.bmm(A, B_t)

#         C12_i = []
#         for i in range(evals_x.size(1)):
#             D12_i = torch.cat([torch.diag(D12[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_x.size(0))], dim=0)
#             C12 = torch.bmm(torch.inverse(A_A_t + self.lambda_ * D12_i), B_A_t[:, i, :].unsqueeze(1).transpose(1, 2))
#             C12_i.append(C12.transpose(1, 2))
#         C12 = torch.cat(C12_i, dim=1)

#         C21_i = []
#         for i in range(evals_y.size(1)):
#             D21_i = torch.cat([torch.diag(D21[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_y.size(0))], dim=0)
#             C21 = torch.bmm(torch.inverse(B_B_t + self.lambda_ * D21_i), A_B_t[:, i, :].unsqueeze(1).transpose(1, 2))
#             C21_i.append(C21.transpose(1, 2))
#         C21 = torch.cat(C21_i, dim=1)

#         return [C12, C21]

class RegularizedCFMNet(nn.Module):
    """Compute the complex functional map matrix representation."""

    def __init__(self, lambda_=1e-3, resolvant_gamma=0.5):
        super().__init__()
        self.lambda_ = lambda_
        self.resolvant_gamma = resolvant_gamma

    def forward(self, feat_x, feat_y, spec_grad_x, spec_grad_y, cevals_x, cevals_y):
        # compute linear operator matrix representation C1 and C2
        cty = torch.complex128
        spec_grad_x, spec_grad_y = spec_grad_x.unsqueeze(0), spec_grad_y.unsqueeze(0)

        F_hat = torch.bmm(spec_grad_x, feat_x.type(cty))
        G_hat = torch.bmm(spec_grad_y, feat_y.type(cty))
        A, B = F_hat, G_hat

        # if normalize input vector fields
        # A, B = A/torch.abs(A), B/torch.abs(B)

        if self.lambda_ == 0:
            Q = (B @ torch.pinverse(A))
            return Q

        # else
        cevals_x, cevals_y = cevals_x.unsqueeze(0), cevals_y.unsqueeze(0)
        D = get_mask(cevals_x.flatten(), cevals_y.flatten(), self.resolvant_gamma, feat_x.device).unsqueeze(0)

        A_t = torch.conj(A.transpose(1, 2))
        A_A_t = torch.bmm(A, A_t)
        B_A_t = torch.bmm(B, A_t)

        Q_i = []
        for i in range(cevals_x.size(1)):
            D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(cevals_x.size(0))], dim=0)
            Q = torch.bmm(torch.inverse(A_A_t + self.lambda_ * D_i),
                          torch.conj(B_A_t[:, i, :].unsqueeze(1).transpose(1, 2)))
            Q_i.append(torch.conj(Q.transpose(1, 2)))
        Q = torch.cat(Q_i, dim=1)

        return Q


from models.dgcnn_sample import DecoderSimpleDGCNN, DecoderSimpleDGCNN_sample
class DGCNNFMNet(nn.Module):
    """
    Compilation of the global model :
    - diffusion net as feature extractor
    - fmap + q-fmap
    - unsupervised loss
    """

    def __init__(self, cfg):
        super().__init__()

        # feature extractor #
        with_grad=True

        self.feature_extractor = DecoderSimpleDGCNN_sample(device=cfg["misc"]["device"])
        # feature extractor #
        # with_grad=True

        # self.feature_extractor = DiffusionNet(
        #      C_in=cfg["fmap"]["C_in"],
        #      C_out=cfg["fmap"]["n_feat"],
        #      C_width=128,
        #      N_block=4,
        #      dropout=True,
        #      with_gradient_features=with_grad,
        #      with_gradient_rotations=with_grad,
        # )
        # regularized fmap
        self.fmreg_net = RegularizedFMNet(lambda_=cfg["fmap"]["lambda_"],
                                          resolvant_gamma=cfg["fmap"]["resolvant_gamma"])
        self.cfmreg_net = RegularizedCFMNet(lambda_=cfg["fmap"]["lambda_"],
                                            resolvant_gamma=cfg["fmap"]["resolvant_gamma"])

        # parameters
        self.n_fmap = cfg["fmap"]["n_fmap"]
        self.n_cfmap = cfg["fmap"]["n_cfmap"]
        self.robust = cfg["fmap"]["robust"]

    def forward(self, batch):
        # verts1, faces1, mass1, L1, evals1, evecs1, gradX1, gradY1 = (batch["shape1"]["xyz"], batch["shape1"]["faces"],
        #                                                              batch["shape1"]["mass"], batch["shape1"]["L"],
        #                                                              batch["shape1"]["evals"], batch["shape1"]["evecs"],
        #                                                              batch["shape1"]["gradX"], batch["shape1"]["gradY"])
        # verts2, faces2, mass2, L2, evals2, evecs2, gradX2, gradY2 = (batch["shape2"]["xyz"], batch["shape2"]["faces"],
        #                                                              batch["shape2"]["mass"], batch["shape2"]["L"],
        #                                                              batch["shape2"]["evals"], batch["shape2"]["evecs"],
        #                                                              batch["shape2"]["gradX"], batch["shape2"]["gradY"])
      
        verts1,  mass1,  evals1, evecs1 = (batch["shape1"]["xyz"],batch["shape1"]["mass"], batch["shape1"]["evals"], batch["shape1"]["evecs"])
        verts2,  mass2, evals2, evecs2 = (batch["shape2"]["xyz"],batch["shape2"]["mass"], batch["shape2"]["evals"], batch["shape2"]["evecs"],)
           
        # print(verts1.shape, verts1.shape)
        # # V1, V2 = batch["shape1"]['partial_verts'], batch["shape2"]['xyz']
        V1_sample, V2_sample = batch["shape1"]['verts_sample'], batch["shape2"]['verts_sample']
        V1_fps_sample, V2_fps_sample = batch["shape1"]['fps'], batch["shape2"]['fps']  
        # # # print(verts1.shape,V1_sample.shape)

        feat1 = self.feature_extractor(verts1.permute(0,2,1), V1_sample.permute(0,2,1),V1_fps_sample)
        feat2 = self.feature_extractor(verts2.permute(0,2,1), V2_sample.permute(0,2,1),V2_fps_sample)
        # # set features to vertices
        # features1, features2 = verts1, verts2
        # print(features1.shape, features2.shape)

        # feat1 = self.feature_extractor(features1, mass1, L=L1, evals=evals1, evecs=evecs1,
                                    #    gradX=gradX1, gradY=gradY1, faces=faces1).unsqueeze(0)
        # feat2 = self.feature_extractor(features2, mass2, L=L2, evals=evals2, evecs=evecs2,
                                    #    gradX=gradX2, gradY=gradY2, faces=faces2).unsqueeze(0)
        # predict fmap
        evecs_trans1, evecs_trans2 =  batch["shape1"]['evecs_trans'], batch["shape2"]['evecs_trans']
        evals1, evals2 = evals1[:,:self.n_fmap], evals2[:,:self.n_fmap]

        #
        C12_pred, C21_pred = self.fmreg_net(feat1, feat2, evals1, evals2, evecs_trans1, evecs_trans2)
        #

        # if we don't have complex spectral info we just return C
        if self.n_cfmap == 0:
            # print(C12_pred.shape, C21_pred.shape, feat1.shape, feat2.shape, evecs_trans1.shape, evecs_trans2.shape)
            # exit()
            return C12_pred, C21_pred, None, feat1, feat2, evecs_trans1, evecs_trans2, evecs1, evecs2

        # # else, also predict cfmap
        # spec_grad1, spec_grad2 = batch["shape1"]["spec_grad"][:self.n_cfmap], batch["shape2"]["spec_grad"][:self.n_cfmap]
        # cevals1, cevals2 = batch["shape1"]["cevals"][:self.n_fmap], batch["shape2"]["cevals"][:self.n_fmap]
        # #

        # cfeat1, cfeat2 = feat1, feat2  # network features
        # Q_pred = self.cfmreg_net(cfeat1, cfeat2, spec_grad1, spec_grad2, cevals1, cevals2)

        # return C12_pred, C21_pred, Q_pred, feat1, feat2, evecs_trans1, evecs_trans2
        pass