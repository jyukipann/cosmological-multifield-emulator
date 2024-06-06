import torch
import torch.nn as nn
import torchvision

class DiscriminatorLoss(nn.Module):
    """
    Focal Loss(BCEの上位互換)とHuber Loss(回帰Loss)を組み合わせる
    """
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.focal_loss = torchvision.ops.focal_loss.sigmoid_focal_loss
        self.huber_loss = torch.nn.HuberLoss()
    
    def forward(self, input, target):
        # 損失の計算：(input - target)の二乗の平均
        return torch.mean((input - target) ** 2)