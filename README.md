# VCL
This is the codebase for our paper "Visual context learning based on cross-modal knowledge for continuous sign language recognition". 
All datasets analysed during this study are available in links. 
## Prerequisites
This project is implemented in Pytorch (>1.8).
ctcdecode==0.4, for beam search decode.
## Data Preparation
PHOENIX-2014 dataset is openly available at [download link](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/).
After finishing dataset download, extract it to ./dataset/phoenix, it is suggested to make a soft link toward downloaded dataset
```ln -s PATH_TO_DATASET/phoenix2014-release ./dataset/phoenix2014```
Run the following command to generate gloss dict.
```cd ./preprocess
python data_preprocess.py --process-image --multiprocessing```
PHOENIX-2014-T dataset is openly available at [download link](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/).
After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.
```ln -s PATH_TO_DATASET/PHOENIX-2014-T-release-v3/PHOENIX-2014-T ./dataset/phoenix2014-T```
Run the following command to generate gloss dict.
```cd ./preprocess
python data_preprocess-T.py --process-image --multiprocessing```
CSL-Daily dataset is openly available at [download link](https://ustc-slr.github.io/datasets/2021_csl_daily/).
After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.
```ln -s PATH_TO_DATASET ./dataset/CSL-Daily```
Run the following command to generate gloss dict.
```cd ./preprocess
python data_preprocess-CSL-Daily.py --process-image --multiprocessing```
