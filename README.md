# cosmological-multifield-emulator
[Cosmological multifield emulator](https://arxiv.org/pdf/2402.10997)
をPyTorchで実装し、検証する。


## 環境構築
```bash
docker-compose up
```


## データセットのダウンロード
```bash
cd {dataset_dir}
wget -l1 -A npy,txt https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/
```

## データセットの命名規則について

[データセットの命名規則](https://camels-multifield-dataset.readthedocs.io/en/latest/data.html)を参照。  
`Maps_prefix_suite_set_z=0.00.npy`を標準とする。  
`prefix`の選択肢  

Gas density

Mgas

16,785

16,380

16,380

–


Gas velocity

Vgas

16,785

16,380

16,380

–

km/s

Gas temperature

T

16,785

16,380

16,380

–

Kelvin

Gas pressure

P

16,785

16,380

16,380

–


Gas metallicity

Z

16,785

16,380

16,380

–

dimensionless

Neutral hydrogen density

HI

16,785

16,380

16,380

–


Electron number density

ne

16,785

16,380

16,380

–


Magnetic fields

B

16,785

–

–

–

Gauss

Magnesium over Iron

MgFe

16,785

16,380

16,380

–

dimensionless

Dark matter density

Mcdm

16,785

16,380

16,380

–


Dark matter velocity

Vcdm

16,785

16,380

16,380

–

km/s

Stellar mass density

Mstar

16,785

16,380

16,380

–


Total matter density

Mtot

16,785

16,380

16,380

49,140


Total

218,205

196,560

196,560

49,140

