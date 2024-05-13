import pathlib
import numpy as np

def split_maps(map_path:pathlib.Path | str, split_index:int):
    maps = np.load(map_path)
    # maps0, maps1 = 