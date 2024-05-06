import torch
import torch.nn as nn

class Generator(nn.Module):
  def __init__(self, z_dim, img_dim):
    super().__init__()
    self.fc1 = nn.Linear(z_dim, 256)
    self.fc2 = nn.Linear(256, 512)
    self.fc3 = nn.Linear(512, img_dim)

  def forward(self, z):
    out = self.fc1(z)
    out = nn.ReLU()(out)
    out = self.fc2(out)
    out = nn.ReLU()(out)
    out = self.fc3(out)
    return out

class Discriminator(nn.Module):
  def __init__(self, img_dim):
    super().__init__()
    self.fc1 = nn.Linear(img_dim, 512)
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, 1)

  def forward(self, img):
    out = self.fc1(img)
    out = nn.ReLU()(out)
    out = self.fc2(out)
    out = nn.ReLU()(out)
    out = self.fc3(out)
    return out
