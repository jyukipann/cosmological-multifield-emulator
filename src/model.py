import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torchvision

class Generator(nn.Module):
    def __init__(self, input_size:tuple, output_size:tuple)->None:
        super().__init__()
        self.input_size = input_size # expected (1, 2, 2)
        self.output_size = output_size # expected (3, 256, 256)
        self.relu = nn.ReLU()
        self.upsampling_layers = [
            nn.ConvTranspose2d(1, 3, 2, stride=2, dtype=torch.float64),
            self.relu,
            nn.ConvTranspose2d(3, 15, 2, stride=2, dtype=torch.float64),
            self.relu,
            nn.ConvTranspose2d(15, 15, 2, stride=2, dtype=torch.float64),
            self.relu,
            nn.ConvTranspose2d(15, 5, 2, stride=2, dtype=torch.float64),
            self.relu,
            nn.ConvTranspose2d(5, 5, 2, stride=2, dtype=torch.float64),
            self.relu,
            nn.ConvTranspose2d(5, 5, 2, stride=2, dtype=torch.float64),
            self.relu,
            nn.ConvTranspose2d(5, 3, 2, stride=2, dtype=torch.float64),
        ]
        self.upsampling_layers = nn.Sequential(*self.upsampling_layers)
        
        
    def forward(self, x):
        x = self.upsampling_layers(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_size:tuple)->None:
        super().__init__()
        self.input_size = input_size # (3, 256, 256)
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads.head = nn.Linear(
            in_features=768, out_features=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        # 形状合わせて無理やり入力
        self.resize = torchvision.transforms.Resize((224,224))
        
    def forward(self, maps):
        x = maps
        x = self.resize(x).to(torch.float)
        x = self.vit(x)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    import torch
    g = Generator((1, 4), (1, 3, 256, 256))
    x = torch.rand((1, 1, 2, 2), dtype=torch.float64)
    print(g.parameters())
    # print(x.shape)
    x = g(x)
    # print(x.shape)
    
    d = Discriminator((1,3,256,256))
    # print(d.vit)
    
    x = d(x)
    print(x.shape)