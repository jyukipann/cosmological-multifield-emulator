from typing import Tuple
import torch
import torch.nn as nn
import torchvision
from torch import tensor
from torch.nn import Sequential as Seq

class Generator(nn.Module):
    def __init__(self, input_size:tuple, output_size:tuple)->None:
        super().__init__()
        self.input_size = input_size # expected (256, 1, 1)
        self.output_size = output_size # expected (3, 256, 256)
        
        max_channels = 1024
        self.init = Seq(
            nn.ConvTranspose2d(
                self.input_size[0], max_channels*2, 4, 1, 0, bias=False),
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
    block  = nn.ModuleList()
    for i in range(times):
        block.append(nn.Conv2d(in_channels if i == 0 else out_channels, out_channels*2, 3, 1, 1, bias=False))
        block.append(nn.BatchNorm2d(out_channels*2))
        if noise_injection:
            block.append(NoiseInjection())
        block.append(nn.GLU(1))
    return Seq(nn.Upsample(scale_factor=2, mode='nearest'), *block)

class SkipLayerExcitation(nn.Module):
    def __init__(
            self, 
            in_channels:int, 
            hidden_channels:int,
            out_res_channels:int,):
        
        super().__init__()
        self.layers = Seq(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(in_channels, hidden_channels, 4, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_res_channels, 1, 1, 0),
            nn.Sigmoid(),
        )
    
    def forward(self, x1: tensor, x2:tensor)->tensor:
        x1 = self.layers(x1)
        return torch.mul(x2, x1)

class Discriminator(nn.Module):
    def __init__(self, input_size:tuple)->None:
        super().__init__()
        self.input_size = input_size # (3, 256, 256)

        channels = [128, 64, 32, 16, 8, 1]

        self.leakyrelu = nn.LeakyReLU(0.2)

        self.downconv128 = nn.Conv2d(self.input_size[0], channels[0], 3, 2, 1)
        self.downconv64 = nn.Conv2d(channels[0], channels[1], 3, 2, 1)
        
        self.downconv32 = nn.Conv2d(channels[1], channels[2], 3, 2, 1)
        self.se32 = SkipLayerExcitation(self.input_size[0], channels[2], channels[2])

        self.downconv16 = nn.Conv2d(channels[2], channels[3], 3, 2, 1)
        self.se16 = SkipLayerExcitation(channels[0], channels[3], channels[3])
        
        self.downconv8 = nn.Conv2d(channels[3], channels[4], 3, 2, 1)
        self.se8 = SkipLayerExcitation(channels[1], channels[4], channels[4])

        self.downconv8_5 = Seq(
            nn.Conv2d(channels[4], channels[5], 3, 1, 1),
            nn.Conv2d(channels[5], channels[5], 4, 1, 0),
            nn.Conv2d(channels[5], 1, 1, 1, 0),
            nn.Flatten()
        )

        self.convolution_low_8 = Seq(
            nn.Conv2d(self.input_size[0], channels[1], 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels[1], channels[2], 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels[2], channels[3], 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels[3], channels[4], 3, 2, 1),
            nn.LeakyReLU(0.2)
        )

        self.convolution_low_5 = Seq(
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

        self.decoder_high_8 = SimpleDecoder(channels[4])
        self.decoder_high_16 = SimpleDecoder(channels[3])
        self.decoder_low = SimpleDecoder(channels[4])
            
        
    def forward(self, high_res, low_res):
        feat128 = self.downconv128(high_res)
        feat128 = self.leakyrelu(feat128)

        feat64 = self.downconv64(feat128)
        feat64 = self.leakyrelu(feat64)

        feat32 = self.downconv32(feat64)
        feat32 = self.leakyrelu(feat32)
        feat32 = self.se32(high_res, feat32)

        feat16 = self.downconv16(feat32)
        feat16 = self.leakyrelu(feat16)
        feat16 = self.se16(feat128, feat16)

        feat8 = self.downconv8(feat16)
        feat8 = self.leakyrelu(feat8)
        feat8 = self.se8(feat64, feat8)

        feat5_high_res = self.downconv8_5(feat8)
        feat8_low_res = self.convolution_low_8(low_res)
        feat5_low_res = self.convolution_low_5(feat8_low_res)

        x = torch.cat((feat5_high_res, feat5_low_res), -1)
        x = self.linear(x)

        deco_high_8 = self.decoder_high_8(feat8)
        deco_high_16 = self.decoder_high_16(feat16)
        deco_low = self.decoder_low(feat8_low_res)

        return x, [deco_high_8, deco_high_16, deco_low]


class SimpleDecoder(nn.Module):
    def __init__(self, input_size:tuple)->None:
        super().__init__()
        self.input_size = input_size # 8 or 16

        channels = [48, 24, 12, 6, 3]

        # 16*16*16を48*8*8にする
        self.pretreatment_for16 = nn.Conv2d(self.input_size, channels[0], 3, 2, 1)
        # 8*8*8を48*8*8にする
        self.pretreatment_for8 = nn.Conv2d(self.input_size, channels[0], 3, 1, 1)
        
        self.SD = Seq(
            upBlock(channels[0], channels[1]), # 1回目
            upBlock(channels[1], channels[2]), # 2回目
            upBlock(channels[2], channels[3]), # 3回目
            upBlock(channels[3], channels[4]), # 4回目
            # nn.Conv2d(channels[4], channels[5], 3, 1, 1),   # (32, 128, 128)を(3, 128, 128)にする
            nn.Tanh()   # 正規化
        )

    def forward(self, input):
        if input.shape[1] == 16:
            feat8 = self.pretreatment_for16(input)
        else:
            feat8 = self.pretreatment_for8(input)

        return self.SD(feat8)


def upBlock(in_channel, out_channel):
    block = Seq(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_channel, out_channel*2, 3, 1, 1),
        nn.BatchNorm2d(out_channel*2),
        nn.GLU(1)
    )
    return block


if __name__ == '__main__':
    import torch
    batch_size = 1
    input_noise_shape = (batch_size, 256, 1, 1)
    output_map_shape = (batch_size, 3, 256, 256)
    noise = torch.rand(input_noise_shape, dtype=torch.float32)
    g = Generator(input_noise_shape[1:], output_map_shape[1:]).eval()
    g = torch.jit.trace(g, (noise,))
    print(g)
    
    
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


    d = Discriminator(output_map_shape[1:]).eval()
    d = torch.jit.trace(d, (high_res, low_res))
    print(d)
    
    x = d(high_res, low_res)
    print(x[0].shape)
    for i in range(3):
        print(x[1][i].shape)    # Decoderで生成した画像のshapeを表示

    torch.functional.ma