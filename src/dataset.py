import torch
import numpy as np
from torch.utils.data import Dataset

class CAMELSMultifieldDataset(Dataset):
    def __init__(self, data_paths:list, params_paths:list, device:torch.device) -> None:
        super().__init__()
        self.maps = [[np.load(p) for p in paths] for paths in data_paths]
        self.params = [np.loadtxt(p) for p in params_paths]
        self.map_lengths = [m[0].shape[0] for m in self.maps]
        self.device = device
        
    def __len__(self) -> int:
        return sum(self.map_lengths)
    
    def __getitem__(self, index) -> None:
        target_map_index = 0
        length_sum = 0
        map_index = 0
        for i, length in enumerate(self.map_lengths):
            if index < length_sum + length:
                target_map_index = i
                map_index = index - length_sum
                break
            
            length_sum += length
        else:
            raise IndexError("CAMELSMultifieldDataset index out of range")
        maps = [m[map_index] for m in self.maps[target_map_index]]
        maps = torch.tensor(maps, dtype=torch.float64, device=self.device)
        params = torch.tensor(self.params[target_map_index][[map_index//15]], device=self.device)
        return maps, params


if __name__ == '__main__':
    data_paths = [
        [
            'dataset/Maps_Mgas_IllustrisTNG_CV_z=0.00.npy',
            'dataset/Maps_HI_IllustrisTNG_CV_z=0.00.npy',
            'dataset/Maps_B_IllustrisTNG_CV_z=0.00.npy',
        ],
    ]
    params_paths = [
        'dataset/params_CV_IllustrisTNG.txt',
    ]
    device = torch.device('cuda')
    cmd = CAMELSMultifieldDataset(data_paths, params_paths, device)
    print(len(cmd))
    maps , params = cmd[0]
    print(maps.shape)
    print(params.shape)
    
    
    