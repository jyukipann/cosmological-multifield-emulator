import torch.nn as nn
import torchvision
from torch import tensor

class ReconstructionLoss(nn.Module):
    def __init__(self) -> None:
        super(ReconstructionLoss, self).__init__()
        self.huber_loss = nn.HuberLoss(reduction='mean')

    def forward(self, input, target) -> tensor:
        return self.huber_loss(input, target)

class FocalLoss(nn.Module):
    def __init__(self) -> None:
        super(FocalLoss, self).__init__()
        self.forcal_loss = torchvision.ops.sigmoid_focal_loss

    def forward(self, input, target) -> tensor:
        return self.forcal_loss(input, target, reduction='mean')

class HingeLoss(nn.Module):
    def __init__(self) -> None:
        super(HingeLoss, self).__init__()
    
    def forward(self, input, is_fake = True) -> tensor:
        if is_fake:
            loss = (1. + input).relu().mean()
        else:
            loss = (1. - input).relu().mean()
        return loss

class WassersteinLoss(nn.Module):
    def __init__(self) -> None:
        super(WassersteinLoss, self).__init__()

    def forward(self, input, is_fake = True) -> tensor:
        loss = input.mean()
        if is_fake:
            loss *= -1
        return loss

if __name__ == '__main__':
    rec_loss_func = ReconstructionLoss()
    focal_loss_func = FocalLoss()
    
    import torch
    import random
    
    # inputs = torch.tensor([0.5,0.5])
    # targets = torch.tensor([0.56, 0.54])
    # rec_loss = rec_loss_func(inputs, targets)
    # print(rec_loss)

    dims = [10, 100, 999]
    for n in dims:
        inputs = torch.rand((n, n)) * 2 - 1
        targets = torch.rand((n, n)) * 2 - 1
        rec_loss = rec_loss_func(inputs, targets)
        print(f'{n=:3}:{rec_loss}')

    for n in dims:
        inputs = torch.rand((n,))
        targets = torch.tensor(random.choices([0.0, 1.0], k=n))
        focal_loss = focal_loss_func(inputs, targets)
        print(f'{n=:3}:{focal_loss}')
