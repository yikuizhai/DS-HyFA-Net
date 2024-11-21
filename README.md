# <p align="center">DS-HyFA-Net: A Deeply Supervised Hybrid Feature Aggregation Network With Multiencoders for Change Detection in High-Resolution Imagery < br > (IEEE TGRS 2024)</p>

This is the official repository for “[DS-HyFA-Net: A Deeply Supervised Hybrid Feature Aggregation Network With Multiencoders for Change Detection in High-Resolution Imagery](https://doi.org/10.1109/TGRS.2024.3471075)”

The repo is based on [CDLab](https://github.com/Bobholamovic/CDLab).

## Abstract

With the advancement of deep learning (DL) technologies, remarkable progress has been achieved in change detection (CD). Existing DL-based methods primarily focus on the discrepancy in bitemporal images, while overlooking the commonality in bitemporal images. However, one of the reasons hindering the improvement of CD performance is the inadequate utilization of image information. To address the above issue, we propose a Deeply Supervised Hybrid Feature Aggregation Network (DS-HyFA-Net). This network predicts changes by integrating the distinctness and the commonality in bitemporal images. Specifically, the DS-HyFA-Net primarily consists of a set of encoders and a Hybrid Feature Aggregation (HyFA) module. It uses a Siamese encoder (or Encoder I) and a specialized encoder (or Encoder II) to extract distinct and common features (CFs) in bitemporal images, respectively. The HyFA module efficiently aggregates distinct and common features (or hybrid features) and generates a change map using a predictor. In addition, a common feature learning strategy (CFLS) is introduced, based on deeply supervised (DS) techniques, to guide Encoder II in learning CFs. Experimental results on three well-recognized datasets demonstrate the effectiveness of the innovative DS-HyFA-Net, achieving F1-Scores of 93.33% on WHU-CD, 90.98% on LEVIR-CD, and 81.14% on SYSU-CD.

## Requirements
- Ubuntu 18.04
- Python 3.7
- Pytorch==1.13.1
- torchvision==0.14.1
  
See [environment.yaml](https://github.com/yikuizhai/DS-HyFA-Net/blob/main/environment.yaml) for details.

## Getting Started

### Prepare the data
 - Prepare datasets into the following structure and change the dataset locations to your own in `src/constants.py`.
   ```
    ├─Train
        ├─A        ...png
        ├─B        ...png
        ├─label    ...png
    ├─Val
        ├─A
        ├─B
        ├─label
    ├─Test
        ├─A
        ├─B
        ├─label
    ```
   
### Train

```
python train.py train --exp_config PATH_TO_CONFIG_FILE --exp_dir PATH_TO_EXPERIMENT
```

### Test

```
python train.py eval --exp_config PATH_TO_CONFIG_FILE --resume PATH_TO_CHECKPOINT --save_on --subset test
```

## Citation

Please cite this if you want to use it in your work.

```
@ARTICLE{DS-HyFA-Net,
  author={Ying, Zilu and Xian, Tingfeng and Zhai, Yikui and Jia, Xudong and Zhang, Hongsheng and Pan, Jiahao and Coscia, Pasquale and Genovese, Angelo and Piuri, Vincenzo and Scotti, Fabio},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={DS-HyFA-Net: A Deeply Supervised Hybrid Feature Aggregation Network With Multiencoders for Change Detection in High-Resolution Imagery}, 
  year={2024},
  volume={62},
  number={},
  pages={1-17},
  keywords={Feature extraction;Data mining;Representation learning;Remote sensing;Transformers;Support vector machines;Manuals;Classification algorithms;Principal component analysis;Faces;Change detection (CD);common feature learning strategy (CFLS);deeply supervised (DS);hybrid feature aggregation (HyFA) module;multiencoder},
  doi={10.1109/TGRS.2024.3471075}}
```
