import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_size:tuple, output_size:tuple)->None:
        super().__init__()
        self.input_size = input_size # expected (b, 4)
        self.output_size = output_size # expected (b, 3, 256, 256)
        self.upsampling = nn.ConvTranspose2d(1, 3, 2, stride=2, dtype=torch.float64)
        
    def forward(self, x):
        x = self.upsampling(x)
        return x

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


if __name__ == '__main__':
    import torch
    g = Generator((1, 4), (1, 3, 256, 256))
    x = torch.rand((1, 4), dtype=torch.float64).reshape((1, 1, 2, 2))
    print(x.shape)
    x = g(x)
    print(x.shape)
    