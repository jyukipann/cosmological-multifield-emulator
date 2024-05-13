import numpy as np
import pathlib
import matplotlib.pyplot as plt

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
    
if __name__ == '__main__':
    print('playground')

    display_maps_test()
    
    # map_counter()