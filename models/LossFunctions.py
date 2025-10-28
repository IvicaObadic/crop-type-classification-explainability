"""
Credits to  github.com/clcarwin/focal_loss_pytorch and
            https://github.com/VSainteuf/pytorch-psetae
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, logpt, target):

        target = target.view(-1, 1)

        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class WeightedMSE(nn.Module):
    def __init__(self):
        super(WeightedMSE, self).__init__()

    def forward(self, ndvipt, ndvi_target):
        weights = torch.var(ndvipt, dim=0)
        wse_tot = weights * (ndvipt - ndvi_target) ** 2
        return torch.mean(wse_tot)


class CosineLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(CosineLoss, self).__init__()
        self.eps = eps

    def forward(self, h_emb):  
        # taken from https://github.com/VITA-Group/Diverse-ViT/blob/main/reg.py        
        # h_emb (B, T, heads)
        # normalize
        target_h_emb = h_emb
        hshape = target_h_emb.shape 
        target_h_emb = target_h_emb.reshape(hshape[0], hshape[1], -1)
        a_n = target_h_emb.norm(dim=2).unsqueeze(2)
        a_norm = target_h_emb / torch.max(a_n, self.eps * torch.ones_like(a_n))

        # patch-wise absolute value of cosine similarity
        sim_matrix = torch.einsum('abc,acd->abd', a_norm, a_norm.transpose(1,2))
        loss_cos = sim_matrix.mean()

        return loss_cos



class CombinedLoss(nn.Module):
    def __init__(self, fn='WCE', class_weights = None, gamma=0, weight_focal=0.8, alpha=None, size_average=True, eps=1e-8):
        super(CombinedLoss, self).__init__()
        print(f'Used loss function: {fn}, gamma: {gamma}')
        
        if fn == 'WCE':
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        elif fn == 'FL':
            self.loss_fn = FocalLoss(gamma=gamma, alpha=alpha, size_average=size_average)
        
        # self.mse = nn.MSELoss()
        self.cos_loss = CosineLoss()
        self.w1 = weight_focal
        self.w2 = 1.0 - weight_focal      # rsme weight
        self.eps = eps
        self.fn = fn
        
    def forward(self, logpt, target, ndvi_pred, ndvi_target, attn_heads):
        loss = self.loss_fn(logpt, target)
        # ndvi_loss = torch.sqrt(self.mse(ndvi_pred, ndvi_target) + self.eps)

        if self.w1 == 1.0:
            return loss
        else:
            attn_loss = self.cos_loss(attn_heads)
            return self.w1*loss + self.w2*attn_loss
    

# Example usage
if __name__ == "__main__":
    logpt = torch.randn((10, 2), requires_grad=True)  # example input logits
    target = torch.randint(0, 2, (10,))#, requires_grad=True)  # example targets
    ndvipt = torch.randn((5, 10), requires_grad=True)  # example NDVI predictions
    ndvi_target = torch.randn((5, 10), requires_grad=True)  # example NDVI targets

    combined_loss = CombinedLoss()
    loss = combined_loss(logpt, target, ndvipt, ndvi_target)
    loss.backward()
    print(loss)