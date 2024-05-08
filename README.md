# cosmological-multifield-emulator
[Cosmological multifield emulator](https://arxiv.org/pdf/2402.10997)
をPyTorchで実装し、検証する。

## 環境構築
```bash
docker-compose up
```

## データセットのダウンロード
powershellではwgetが標準でなかったため、wsl経由のbashでwgetした。
```bash
cd {dataset_dir}
wget -r -l1 -A npy,txt https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/
```
```txt
FINISHED --2024-05-06 21:52:24--
Total wall clock time: 7h 57m 26s
Downloaded: 201 files, 162G in 7h 56m 39s (5.80 MB/s)
```
約8時間かかった。

## データセットついて

[データセットの説明](https://camels-multifield-dataset.readthedocs.io/en/latest/data.html)を参照。  
命名規則は`Maps_prefix_suite_set_z=0.00.npy`を標準とする。

### `prefix` の選択肢
Fieldの選択をprefixにより行う。

今回の実験では、`Mgas`(Gass Ddensity)と`HI`(Neutral hydrogen density)と`B`(Magnetic fields)をデータセットとして用いる。

| Field                    | Prefix  |
| ------------------------ | ------- |
| Gas density              | `Mgas`  |
| Gas velocity             | `Vgas`  |
| Gas temperature          | `T`     |
| Gas pressure             | `P`     |
| Gas metallicity          | `Z`     |
| Neutral hydrogen density | `HI`    |
| Electron number density  | `ne`    |
| Magnetic fields          | `B`     |
| Magnesium over Iron      | `MgFe`  |
| Dark matter density      | `Mcdm`  |
| Dark matter velocity     | `Vcdm`  |
| Stellar mass density     | `Mstar` |
| Total matter density     | `Mtot`  |

### `suite` の選択肢
	
Magnetic fieldsは`IllustrisTNG`のみのデータとなるため、`IllustrisTNG`のみを使用する。
- `IllustrisTNG`
- `SIMBA`
- `Astrid`
- `Nbody_IllustrisTNG`
- `Nbody_SIMBA`
- `Nbody_Astrid`

#### 各`suite`の説明
CMD has been generated from thousands of state-of-the-art (magneto-)hydrodynamic and gravity-only N-body simulations from the CAMELS project. CMD data can be classified into different suites, that indicate the type of simulation used to create the data:

CMD は、CAMELS プロジェクトによる何千もの最先端の (磁気) 流体力学および重力のみの N 体シミュレーションから生成されました。 CMD データは、データの作成に使用されるシミュレーションのタイプを示すさまざまなスイートに分類できます。

- `IllustrisTNG`. These magneto-hydrodynamic simulations follow the evolution of gas, dark matter, stars, and black-holes. They also simulate magnetic fields. CMD uses 1,088 of these simulations.
  - これらの磁気流体力学シミュレーションは、ガス、暗黒物質、星、ブラックホールの進化を追跡します。また、磁場もシミュレートします。 CMD はこれらのシミュレーションのうち 1,088 を使用します。

- `SIMBA`. These hydrodynamic simulations follow the evolution of gas, dark matter, stars, and black-holes. CMD uses 1,088 of these simulations.
  - これらの流体力学シミュレーションは、ガス、暗黒物質、星、ブラックホールの進化を追跡します。 CMD はこれらのシミュレーションのうち 1,088 を使用します。

- `Astrid`. These hydrodynamic simulations follow the evolution of gas, dark matter, stars, and black-holes. CMD uses 1,088 of these simulations.
  - これらの流体力学シミュレーションは、ガス、暗黒物質、星、ブラックホールの進化を追跡します。 CMD はこれらのシミュレーションのうち 1,088 を使用します。

- `N-body`. These gravity-only N-body simulation only follow the evolution of dark matter. Thus, they do not model astrophysical processes such as the formation of stars and the feedback from black-holes. There is an N-body simulation for each (magneto-)hydrodynamic simulation. CMD uses 2,000 of these simulations.
  - これらの重力のみの N 体シミュレーションは、暗黒物質の進化のみを追跡します。したがって、星の形成やブラックホールからのフィードバックなどの天体物理学的プロセスはモデル化されていません。 (磁気) 流体力学シミュレーションごとに N 体シミュレーションがあります。 CMD はこれらのシミュレーションを 2,000 使用します。

### `set` の選択肢
- `1P`
- `CV`
- `LH`

#### 各`sets`の説明
Each suite contains different sets, that indicate how the value of the labels of the underlying simulations are organized:

各スイートには、基礎となるシミュレーションのラベルの値がどのように構成されているかを示す異なるセットが含まれています。

- `CV`. The value of the labels is always the same and correspond to the fiducial model. The 2D maps and 3D grids only differ on the initial conditions of the simulations run. This set contains 27 simulations.
  - ラベルの値は常に同じであり、基準モデルに対応します。 2D マップと 3D グリッドは、実行されるシミュレーションの初期条件においてのみ異なります。このセットには 27 のシミュレーションが含まれています。

- `1P`. The value of the labels is varied one-at-a-time. I.e. the 2D maps and 3D grids have labels whose value only differ in one element from the value of the fiducial maps (CV set). In this case, the initial conditions are always the same. This set contains 61 simulations.
  - ラベルの値は一度に 1 つずつ変更されます。つまり、 2D マップと 3D グリッドにはラベルがあり、その値は基準マップ (CV セット) の値と 1 つの要素のみが異なります。この場合、初期条件は常に同じです。このセットには 61 のシミュレーションが含まれています。

- `LH`. The value of all labels is different in each simulation and the values are organized in a latin-hypercube. The value of the initial conditions is different in each simulation. This set contains 1,000 simulations.
  - すべてのラベルの値はシミュレーションごとに異なり、値はラテンハイパーキューブに編成されます。初期条件の値はシミュレーションごとに異なります。このセットには 1,000 のシミュレーションが含まれています。

- `EX`. The value of the labels is chosen to be extreme and the initial conditions of the simulations are the same. This set contains 4 simulations.
  - ラベルの値は極端になるように選択され、シミュレーションの初期条件は同じです。このセットには 4 つのシミュレーションが含まれています。

- `BE`. The underlying simulations have the same initial conditions and the same value of the labels (the fiducial ones). The only difference between the simulations is due to random noise from numerical approximations. This set contains 27 simulations. So far, this set is only present for the IllutrisTNG suite.
  - 基礎となるシミュレーションには、同じ初期条件と同じラベル (基準ラベル) の値があります。シミュレーション間の唯一の違いは、数値近似によるランダム ノイズによるものです。このセットには 27 のシミュレーションが含まれています。今のところ、このセットは IllutrisTNG スイートにのみ存在します。

### 各フィールドのパラメータ
`.txt`にある。

## データセットの扱い方と表示
`.npy`形式のファイルの中身は`(n, 256, 256)`のndarrayだった（少なくとも、`Mgas`, `HI`, `B`については）。  
1次元目で１枚取り出して、対数を取って適当なカラーマップを用いてプロットすることで、論文のような画像が作れる。  
![Mgas data](dump/first_plot_Mgas.png)

