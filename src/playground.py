from typing import Tuple
import utils
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import model
import loss
import dataset
import torchvision

def display_maps_test():
    out_dir = pathlib.Path("dump/")
    out_dir.mkdir(parents=True, exist_ok=True)    
    
    def load_map0_and_plot(fmaps, cmap=plt.cm.hot, log=True):
        maps = np.load(fmaps)
        print(maps.shape)
        plt.clf()
        if log:
            maps[0] = np.log10(maps[0])
        plt.pcolor(maps[0], cmap=cmap)
        plt.colorbar()
    
    fmaps = 'dataset/Maps_Mgas_IllustrisTNG_CV_z=0.00.npy'
    load_map0_and_plot(fmaps=fmaps, cmap=plt.cm.hot)
    plt.savefig(out_dir/"first_plot_Mgas.png")
    
    fmaps = 'dataset/Maps_HI_IllustrisTNG_CV_z=0.00.npy'
    load_map0_and_plot(fmaps=fmaps, cmap=plt.cm.Greens)
    plt.savefig(out_dir/"first_plot_HI.png")
    
    fmaps = 'dataset/Maps_B_IllustrisTNG_CV_z=0.00.npy'
    load_map0_and_plot(fmaps=fmaps, cmap=plt.cm.cividis)
    plt.savefig(out_dir/"first_plot_B.png")

def map_counter():
    map_paths = list(pathlib.Path('dataset').glob('*.npy'))
    print(map_paths)
    
    for path in map_paths:
        print(f"{str(path)}, {np.load(path).shape}")

def tensorboard_test():
    dir = 'dump/tb_test/'
    writer = SummaryWriter(log_dir=dir,)
    g = model.Generator((1, 256, 1, 1), (1, 3, 256, 256))
    for i in range(100):
        writer.add_scalar('accuracy', i/100, i)
        g_img, _ = g(torch.rand((1,256,1,1)))
        mgas,hi,b = g_img[0].detach().numpy()
        mgas = utils.plot_map(mgas, utils.PREFIX_CMAP_DICT['Mgas'])
        hi = utils.plot_map(hi, utils.PREFIX_CMAP_DICT['HI'])
        b = utils.plot_map(b, utils.PREFIX_CMAP_DICT['B'])
        mgas_hi_b = torch.cat([mgas, hi, b], dim=2)
        # print(mgas_hi_b.size())
        # exit()
        writer.add_image("generated_image/mgas_hi_b", mgas_hi_b, i)
        writer.add_image("generated_image/mgas", mgas, i)
        writer.add_image("generated_image/hi", hi, i)
        writer.add_image("generated_image/b", b, i)


def hist_map(map_data:np.ndarray)->Tuple[np.ndarray]:
    return np.histogram(map_data,)

def データの統計量を計算():
    # data = utils.load_from_pickle('dataset/Maps_IllustrisTNG_LH_z=0.00/0.pkl')
    # mgas = data['Mgas']
    # hi = data['HI']
    # b = data['B']
    # hist = np.histogram(mgas,bins=500)
    # print(hist)

    # mean = np.mean(mgas)
    # var = np.var(mgas)
    # print(mean, var)

    # mgas = np.log10(mgas)
    # mean = np.mean(mgas)
    # var = np.var(mgas)
    # print(mean, var)

    # plt.stairs(hist[0], hist[1])
    # plt.yscale('log')
    # plt.savefig('dump/Mgas_0_hist.png')
    
    set_value = 'LH'
    map_paths = [
        f'dataset/Maps_Mgas_IllustrisTNG_{set_value}_z=0.00.npy',
        f'dataset/Maps_HI_IllustrisTNG_{set_value}_z=0.00.npy',
        f'dataset/Maps_B_IllustrisTNG_{set_value}_z=0.00.npy',
    ]
    for i, p  in enumerate(map_paths):
        print(p)
        data_maps = np.load(p)
        data_maps = np.log10(data_maps)
        data_max = np.max(data_maps)
        data_min = np.min(data_maps)
        data_mean = np.mean(data_maps)
        data_var = np.var(data_maps)
        print(f'\t{data_mean=}, {data_var=}')
        print(f'\t{data_min=}, {data_max=}')

def lossの挙動を確認したい():

    dir_path = 'dataset/normalization/Maps_IllustrisTNG_LH_z=0.00'
    cmd = dataset.CAMELSMultifieldDataset(
        dir_path=dir_path,
        ids=list(range(15000)),
    )
    cmd = torch.utils.data.DataLoader(
        cmd,
        batch_size=1, 
        shuffle=False,
        num_workers=1,
    )
    cmd = iter(cmd)

    gen = model.Generator((256,1,1), (3,256,256))
    dis = model.Discriminator((3,256,256))

    rec_loss_func = loss.ReconstructionLoss()
    focal_loss_func = loss.FocalLoss()

    noise = torch.rand((256,1,1))

    batch, params = cmd.__next__()
    _, params_dim = params.size()
    params = torch.reshape(params, (-1, params_dim, 1, 1))
    batch_low_res =  torchvision.transforms.functional.resize(
        batch, (128, 128))

    ret = dis(batch, batch_low_res)
    ret, low_res_rec_maps = ret[0], ret[1:]

    print(f'{low_res_rec_maps[0].max()=}')
    print(f'{low_res_rec_maps[0].min()=}')
    print(f'{batch_low_res.max()=}')
    print(f'{batch_low_res.min()=}')

    rec_loss = 0
    rec_loss += rec_loss_func(low_res_rec_maps[0], batch_low_res)
    rec_loss += rec_loss_func(low_res_rec_maps[1], batch_low_res)
    rec_loss += rec_loss_func(low_res_rec_maps[2], batch_low_res)
    rec_loss /= 3
    print(f'{rec_loss=}')
    focal_loss = focal_loss_func(ret, torch.ones_like(ret))
    print(f'{focal_loss=}')

    # hi_res, low_res = gen(noise)
    # ret = dis(hi_res, low_res)
    # ret, low_res_rec_maps = ret[0], ret[1:]
    

if __name__ == '__main__':
    print('playground')
    # tensorboard_test()
    # display_maps_test()
    # map_counter()
    # データの統計量を計算()

    lossの挙動を確認したい()