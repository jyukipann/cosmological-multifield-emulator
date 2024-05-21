import pathlib
import numpy as np
from typing import List
import pickle
import tqdm
import matplotlib.pyplot as plt
import io
import torchvision
import torch
from torch import tensor

PREFIX_CMAP_DICT = {
    'Mgas':plt.cm.hot,
    'HI':plt.cm.Greens,
    'B':plt.cm.cividis,
}

def save_as_pickle(path:pathlib.Path, obj:any)->None:
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_from_pickle(path:pathlib.Path)->dict:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def extract_map(
        map_paths:List[pathlib.Path | str,],
        params_path:pathlib.Path | str,
        out_dir: pathlib.Path | str,
    )->None:
    
    map_paths = [pathlib.Path(p) for p in map_paths]
    maps = {p.name.split('_')[1]: 
        np.load(p, mmap_mode=print(f'loading {p.name}')) for p in map_paths}
    params = np.loadtxt(params_path)
        
    maps_counts = set((m.shape[0] for m in maps.values()))
    
    # マップに含まれる画像枚数が同じかを検証
    assert len(maps_counts) == 1

    map_count = maps_counts.pop()
    
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for i in tqdm.tqdm(range(map_count)):
        param = params[i//15]
        info = dict(
            id=i, 
            omega_m=param[0], sigma_8=param[1],
            A_SN1=param[2], A_SN2=param[3],
            A_AGN1=param[4], A_AGN2=param[5],
        )
        data = info | {k:v[i] for k, v in maps.items()}
        save_as_pickle(out_dir/f'{i}.pkl', data)
        
def plot_maps(data:dict, out_dir:pathlib.Path | str | None = None)->dict:
    if out_dir is not None:
        out_dir = pathlib.Path(out_dir)
    maps = dict(id=data['id'])
    for prefix, cmap in PREFIX_CMAP_DICT.items():
        map_data = data[prefix]
        maps[prefix] = plot_map(
            map_data, cmap,
            None if out_dir is None else out_dir/f'{data["id"]}_{prefix}.png')
    return maps
    
def plot_map(map_data:tensor, cmap:plt.cm, out_path=None)->tensor:
    h, w = map_data.shape
    plt.gca().clear()
    plt.figure(figsize=(w/100, h/100))
    plt.pcolor(np.log10(map_data), cmap=cmap)
    plt.axis('tight')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if out_path is not None:
        plt.savefig(out_path)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    enc = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    dst = torchvision.io.decode_image(tensor(enc).to(torch.uint8))
    plt.clf()
    plt.close()
    return dst[:-1]
        
if __name__ == '__main__':
    # set_value = 'EX'
    set_value = 'LH'
    map_paths = [
        f'dataset/Maps_Mgas_IllustrisTNG_{set_value}_z=0.00.npy',
        f'dataset/Maps_HI_IllustrisTNG_{set_value}_z=0.00.npy',
        f'dataset/Maps_B_IllustrisTNG_{set_value}_z=0.00.npy',
    ]
    params_path = f'dataset/params_{set_value}_IllustrisTNG.txt'
    out_dir = f'dataset/Maps_IllustrisTNG_{set_value}_z=0.00'
    
    # extract_map(map_paths, params_path, out_dir)
    
    data_id_0 = load_from_pickle(pathlib.Path(out_dir) / f'0.pkl')
    print(data_id_0)
    images = plot_maps(data_id_0, out_dir)
    print(images['Mgas'].shape)
    
    torchvision.io.write_png(images['Mgas'], 'dump/0_Mgas.png')
    
    
    
    
    
    
    