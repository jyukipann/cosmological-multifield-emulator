import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple
import utils
import pathlib

class CAMELSMultifieldDataset(Dataset):
    def __init__(
            self, 
            dir_path: pathlib.Path | str, 
            ids:list,) -> None:
        
        super().__init__()
        self.ids = ids
        self.dir_path = pathlib.Path(dir_path)
        self.prefixs = ('Mgas', 'HI', 'B')
        
    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, index) -> Tuple[torch.tensor, torch.tensor]:
        data = utils.load_from_pickle(self.dir_path/f'{index}.pkl')
        maps = [
            torch.tensor(data[prefix]).unsqueeze(0) for prefix in self.prefixs]
        maps = torch.cat(maps, 0)
        params = torch.tensor([
            data['omega_m'], 
            data['sigma_8'],
            data['A_SN1'],
            data['A_SN2'],
            data['A_AGN1'],
            data['A_AGN2'],
        ])
        return maps, params

if __name__ == '__main__':
    dir_path = 'dataset/Maps_IllustrisTNG_LH_z=0.00'
    train_index_set = list(range(15000))[:10000]
    cmd = CAMELSMultifieldDataset(dir_path, train_index_set)
    print(len(cmd))
    maps , params = cmd[0]
    print(maps.shape)
    print(params.shape)
    
    # dataloader = torch.utils.data.DataLoader(
    #     cmd,
    #     batch_size=2, 
    #     shuffle=True, 
    # )
    
    
    