import torch.nn as nn
import torchvision

class ReconstructionLoss(nn.Module):
    def __init__(self) -> None:
        super(ReconstructionLoss, self).__init__()
        self.huber_loss = nn.HuberLoss(reduction='mean')

    def forward(self, input, target):
        return self.huber_loss(input, target)

class FocalLoss(nn.Module):
    def __init__(self) -> None:
        super(FocalLoss, self).__init__()
        self.forcal_loss = torchvision.ops.sigmoid_focal_loss

    def forward(self, input, target):
        return self.forcal_loss(input, target, reduction='mean')


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
