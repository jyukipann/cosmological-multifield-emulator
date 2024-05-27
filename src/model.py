from typing import Tuple
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torchvision
from torch import tensor
from torch.nn import Sequential as Seq

class Generator(nn.Module):
    def __init__(self, input_size:tuple, output_size:tuple)->None:
        super().__init__()
        self.input_size = input_size # expected (b, 256, 1, 1)
        self.output_size = output_size # expected (3, 256, 256)
        
        max_channels = 1024
        self.init = Seq(
            nn.ConvTranspose2d(
                self.input_size[1], max_channels*2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(max_channels*2),
            nn.GLU(1),
        )
        channels = [max_channels // 2**i for i in [0, 1, 2, 3, 4, 5, 6, 7]]

        self.upconv8 = Up(channels[0], channels[1])
        self.upconv16 = Up(channels[1], channels[2])
        self.upconv32 = Up(channels[2], channels[3])
        self.upconv64 = Up(channels[3], channels[4])
        self.upconv128 = Up(channels[4], channels[5])
        self.upconv256 = Up(channels[5], channels[6])
        
        self.se64 = SkipLayerExcitation(channels[0], channels[4], channels[4])
        self.se128 = SkipLayerExcitation(channels[1], channels[5], channels[5])
        self.se256 = SkipLayerExcitation(channels[2], channels[6], channels[6])
        
        self.arrange_channel128 = nn.Conv2d(channels[5], 3, 3, 1, 1, bias=True)
        self.arrange_channel256 = nn.Conv2d(channels[6], 3, 3, 1, 1, bias=True)
        
        
    def forward(self, x)->Tuple[tensor, tensor]:
        feat4 = self.init(x)
        feat8 = self.upconv8(feat4)
        feat16 = self.upconv16(feat8)
        feat32 = self.upconv32(feat16)
        
        feat64 = self.se64(feat4, self.upconv64(feat32))
        feat128 = self.se128(feat8, self.upconv128(feat64))
        feat256 = self.se256(feat16, self.upconv256(feat128))
        
        low_res = self.arrange_channel128(feat128)
        high_res = self.arrange_channel256(feat256)
        low_res = torch.abs(low_res)
        high_res = torch.abs(high_res)
        return high_res, low_res

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)
        return feat + self.weight * noise

def Up(
        in_channels:int, out_channels:int, 
        times:int=1, noise_injection:bool=False,)->nn.Module:
    block  = []
    for i in range(times):
        block += [
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels*2),
        ]
        if noise_injection:
            block.append(NoiseInjection())
        block.append(nn.GLU(1))
    return Seq(nn.Upsample(scale_factor=2, mode='nearest'), *block)

class SkipLayerExcitation(nn.Module):
    def __init__(
            self, 
            low_res_channels:int, 
            hidden_channels:int,
            high_res_channels:int,):
        
        super().__init__()
        self.layers = Seq(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(low_res_channels, hidden_channels, 4, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_channels, high_res_channels, 1, 1, 0),
            nn.Sigmoid(),
        )
    
    def forward(self, low_res: tensor, high_res:tensor)->tensor:
        return high_res * self.layers(low_res)

class Discriminator(nn.Module):
    def __init__(self, input_size:tuple)->None:
        super().__init__()
        self.input_size = input_size # (3, 256, 256)
        # self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        # self.vit.heads.head = nn.Linear(
        #     in_features=768, out_features=1, bias=True)
        # self.sigmoid = nn.Sigmoid()
        # # 形状合わせて無理やり入力
        # self.resize = torchvision.transforms.Resize((224,224))



        # max_channels = 1024
        channels = [128, 64, 32, 16, 8, 1]
        # channels = [max_channels // 2**i for i in [0, 1, 2, 3, 4, 5, 6, 7]]

        
        self.convolution_high = Seq(
            nn.Conv2d(self.input_size[0], channels[0], 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels[0], channels[1], 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels[1], channels[2], 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels[2], channels[3], 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels[3], channels[4], 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels[4], channels[5], 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels[5], channels[5], 4, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels[5], 1, 1, 1, 0),
            nn.Flatten()
        )


        self.convolution_low = Seq(
            nn.Conv2d(self.input_size[0], channels[1], 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels[1], channels[2], 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels[2], channels[3], 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels[3], channels[4], 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels[4], channels[5], 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels[5], channels[5], 4, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels[5], 1, 1, 1, 0),
            nn.Flatten()
        )

        self.linear = Seq(
            nn.Linear(50, 25),
            nn.Linear(25, 12),
            nn.Linear(12, 1)
        )
        
    
        
    def forward(self, high_res, low_res):
        x = high_res
        x = self.convolution_high(x)
        y = low_res
        y = self.convolution_low(y)

        x = torch.cat((x, y), -1)
        
        x = self.linear(x)

        return x


if __name__ == '__main__':
    import torch
    batch_size = 1
    input_noise_shape = (batch_size, 256, 1, 1)
    output_map_shape = (batch_size, 3, 256, 256)
    g = Generator(input_noise_shape, output_map_shape).eval()
    noise = torch.rand(input_noise_shape, dtype=torch.float32)
    
    # print(g.parameters())
    # print(x.shape)
    high_res, low_res = g(noise)
    print(f"{high_res.size()=} {low_res.size()=}")

    # import utils, datetime
    # now = datetime.datetime.now()
    # utils.plot_map(
    #     high_res[0,0].detach().numpy(), 
    #     utils.PREFIX_CMAP_DICT['Mgas'], 
    #     f'dump/Mgas_{now}.png',
    # )

    
    d = Discriminator((3,256,256))
    # print(d.vit)
    
    x = d(high_res, low_res)
    print(x.shape)

    