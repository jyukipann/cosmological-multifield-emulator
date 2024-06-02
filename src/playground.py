from typing import Tuple
import utils
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import model

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

if __name__ == '__main__':
    print('playground')
    tensorboard_test()
    # display_maps_test()
    # map_counter()