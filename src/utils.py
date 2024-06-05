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

DATA_NORMALIZE_INFO = {
    'Mgas':{'log10_range' : (7, 15),},
    'HI': {'log10_range' : (0, 14),},
    'B': {'log10_range' : (-16, -3),},
}

def save_as_pickle(path:pathlib.Path, obj:any)->None:
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_from_pickle(path:pathlib.Path)->dict:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

"""
これをもとに正規化手法を考える
dataset/Maps_Mgas_IllustrisTNG_LH_z=0.00.npy
        data_mean=10.41468, data_var=0.24149005
        data_min=8.836785, data_max=14.540673
dataset/Maps_HI_IllustrisTNG_LH_z=0.00.npy
        data_mean=4.7845917, data_var=1.0227535
        data_min=0.90377337, data_max=13.424203
dataset/Maps_B_IllustrisTNG_LH_z=0.00.npy
        data_mean=-12.466553, data_var=4.1441007
        data_min=-15.31872, data_max=-3.7545707
"""

def normalize_min_max(x, x_min, x_max):
    return ((2 * x) - x_max - x_min) / (x_max - x_min)
    
def normalize_min_max_reverse(x, x_min, x_max):
    return (x * (x_max - x_min) + x_max + x_min) / 2

def mgas_pipeline(data):
    x = data
    x_min, x_max = DATA_NORMALIZE_INFO['Mgas']['log10_range']
    x = np.log10(x)
    x = normalize_min_max(x, x_min, x_max)
    return x

def hi_pipeline(data):
    x = data
    x_min, x_max = DATA_NORMALIZE_INFO['HI']['log10_range']
    x = np.log10(x)
    x = normalize_min_max(x, x_min, x_max)
    return x

def b_pipeline(data):
    x = data
    x_min, x_max = DATA_NORMALIZE_INFO['B']['log10_range']
    x = np.log10(x)
    x = normalize_min_max(x, x_min, x_max)
    return x

DATA_NORMALIZE_PIPELINE = {
    'Mgas': mgas_pipeline,
    'HI': hi_pipeline,
    'B': b_pipeline,
}

def mgas_pipeline_reverse(data):
    x = data
    x_min, x_max = DATA_NORMALIZE_INFO['Mgas']['log10_range']
    x = normalize_min_max_reverse(x, x_min, x_max)
    x = np.power(10, x)
    return x

def hi_pipeline_reverse(data):
    x = data
    x_min, x_max = DATA_NORMALIZE_INFO['HI']['log10_range']
    x = normalize_min_max_reverse(x, x_min, x_max)
    x = np.power(10, x)
    return x

def b_pipeline_reverse(data):
    x = data
    x_min, x_max = DATA_NORMALIZE_INFO['B']['log10_range']
    x = normalize_min_max_reverse(x, x_min, x_max)
    x = np.power(10, x)
    return x

DATA_NORMALIZE_PIPELINE_REVERSE = {
    'Mgas': mgas_pipeline_reverse,
    'HI': hi_pipeline_reverse,
    'B': b_pipeline_reverse,
}

def extract_map(
        map_paths:List[pathlib.Path | str,],
        params_path:pathlib.Path | str,
        out_dir: pathlib.Path | str,
    )->None:
    
    map_paths = [pathlib.Path(p) for p in map_paths]
    maps = { p.name.split('_')[1]: 
        np.load(p, mmap_mode=print(f'loading {p.name}')) for p in map_paths }
    params = np.loadtxt(params_path)

    # データ正規化
    for k in maps.keys():
        if k in DATA_NORMALIZE_PIPELINE:
            maps[k] = DATA_NORMALIZE_PIPELINE[k](maps[k])
    
    # マップに含まれる画像枚数が同じかを検証
    maps_counts = set((m.shape[0] for m in maps.values()))
    assert len(maps_counts) == 1
    
    # 枚数取得
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


def plot_maps(data:dict, out_dir:pathlib.Path | str | None = None, log10=True)->dict:
    if out_dir is not None:
        out_dir = pathlib.Path(out_dir)
    maps = dict(id=data['id'])
    for prefix, cmap in PREFIX_CMAP_DICT.items():
        map_data = data[prefix]
        maps[prefix] = plot_map(
            map_data, cmap,
            None if out_dir is None else out_dir/f'{data["id"]}_{prefix}.png', 
            log10=log10)
    return maps
    
def plot_map(map_data:tensor, cmap:plt.cm, out_path=None, log10=True)->tensor:
    h, w = map_data.shape
    plt.gca().clear()
    plt.figure(figsize=(w/100, h/100))
    if log10:
        map_data = np.log10(map_data)
    plt.pcolor(map_data, cmap=cmap)
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
    set_value = 'LH'
    map_paths = [
        f'dataset/Maps_Mgas_IllustrisTNG_{set_value}_z=0.00.npy',
        f'dataset/Maps_HI_IllustrisTNG_{set_value}_z=0.00.npy',
        f'dataset/Maps_B_IllustrisTNG_{set_value}_z=0.00.npy',
    ]
    params_path = f'dataset/params_{set_value}_IllustrisTNG.txt'
    out_dir = f'dataset/normalization/Maps_IllustrisTNG_{set_value}_z=0.00'
    
    # extract_map(map_paths, params_path, out_dir)
    
    data_id_0 = load_from_pickle(pathlib.Path(out_dir) / f'0.pkl')
    data_id_0['Mgas'] = DATA_NORMALIZE_PIPELINE_REVERSE['Mgas'](
        data_id_0['Mgas'])
    data_id_0['HI'] = DATA_NORMALIZE_PIPELINE_REVERSE['HI'](
        data_id_0['HI'])
    data_id_0['B'] = DATA_NORMALIZE_PIPELINE_REVERSE['B'](
        data_id_0['B'])
    print(data_id_0)
    images = plot_maps(data_id_0, out_dir)
    print(images['Mgas'].shape)
    
    """
    {
        "id": int,
        "omega_m": float,
        "sigma_8": float,
        "A_SN1": float,
        "A_SN2": float,
        "A_AGN1": float,
        "A_AGN2": float,
        "Mgas": numpy.ndarray,
        "HI": numpy.ndarray,
        "B": numpy.ndarray,
    }
    """