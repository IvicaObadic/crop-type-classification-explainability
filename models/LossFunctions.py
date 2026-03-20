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

    

# Example usage
if __name__ == "__main__":
    logpt = torch.randn((10, 2), requires_grad=True)  # example input logits
    target = torch.randint(0, 2, (10,))#, requires_grad=True)  # example targets
    ndvipt = torch.randn((5, 10), requires_grad=True)  # example NDVI predictions
    ndvi_target = torch.randn((5, 10), requires_grad=True)  # example NDVI targets

    loss = FocalLoss(gamma=1.0)(logpt, target)
    loss.backward()
    print(loss)