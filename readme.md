# A-SATMVSNet: An Attention-Aware Multi-view Stereo Network based on Satellite Imagery

Official Implementation of A-SATMVSNet: An Attention-Aware Multi-view Stereo Network based on Satellite Imagery.

### Requirements

For more details, please refer to environment.yaml. And You can simply import this environment from the yaml file via conda:

`conda env create -f environment.yml`

`conda activate a_satmvsnet`

Some packages are list here:

| package        | version  |
| -------------- | -------- |
| gdal           | 3.3.1    |
| matplotlib     | 3.4.3    |
| numpy          | 1.12.5   |
| tensorboardx   | 2.5      |
| pytorch        | 1.4.0    |
| torchvision    | 0.5.0    |
| numpy-groupies | 0.9.14   |
| opencv-python  | 4.5.5.62 |

### Data Preparation
See [WHU_TLC/readme.md](WHU_TLC/readme.md) for more details. And rename the "open_dataset"  to "open_dataset_rpc".

Train and test, both of them are needed to fill the default values.

### Train
Train on WHU-TLC dataset using RPC warping:

`python train.py`

Train on WHU-TLC dataset using homography warping:

`python train.py `

### Predict
If you want to predict your own dataset, you need to If you want to predict on your own dataset, you need to first organize your dataset into a folder similar to the WHU-TLC dataset. And then run:

`python predict.py`
