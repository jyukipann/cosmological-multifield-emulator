import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class Generator(nn.Module):
    def __init__(self, input_size:tuple, output_size:tuple)->None:
        super().__init__()
        self.input_size = input_size # expected (b, 1, 2, 2)
        self.output_size = output_size # expected (b, 3, 256, 256)
        self.upsampling_layers = [
            nn.ConvTranspose2d(1, 3, 2, stride=2, dtype=torch.float64),
            nn.ConvTranspose2d(3, 15, 2, stride=2, dtype=torch.float64),
            nn.ConvTranspose2d(15, 15, 2, stride=2, dtype=torch.float64),
            nn.ConvTranspose2d(15, 5, 2, stride=2, dtype=torch.float64),
            nn.ConvTranspose2d(5, 5, 2, stride=2, dtype=torch.float64),
            nn.ConvTranspose2d(5, 5, 2, stride=2, dtype=torch.float64),
            nn.ConvTranspose2d(5, 3, 2, stride=2, dtype=torch.float64),
        ]
        self.relu = nn.ReLU()
        
    def forward(self, x):
        for upsampling in self.upsampling_layers:
            x = upsampling(x)
            x = self.relu(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_size:tuple)->None:
        super().__init__()
        self.input_size = input_size
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads.head = nn.Linear(
            in_features=768, out_features=1, bias=True)
        

    def forward(self, maps):
        x = self.vit(maps)
        return x


if __name__ == '__main__':
    import torch
    g = Generator((1, 4), (1, 3, 256, 256))
    x = torch.rand((1, 1, 2, 2), dtype=torch.float64)
    # print(x.shape)
    x = g(x)
    # print(x.shape)
    
    d = Discriminator((1,3,256,256))
    # print(d.vit)
    
    x = d(x)
    print(x.shape)