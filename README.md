## [Appearance-Preserving 3D Convolution for Video-based Person Re-identification](http://arxiv.org/abs/2007.08434)

#### Requirements: Python=3.6 and Pytorch=1.0.0
#### 这篇文章，主要使用的方法是，AP3D来提取连续视频的行人重识别的特征，其中主要是分为两个步骤，APM主要是来解决邻居帧之间外观帧不
#### 匹配的现象的，因为单纯使用3D卷积是会破坏外观特征的提取的过程的。其中我也是对这篇论文，进行大量的注释，希望能够帮到在学习的你
#### 因为，在我最开始的学习的时候，都是没有注释的，所当我开始的时候是非常困难。当然注释中肯定还是会出现错误的，希望你使用embed包
#### 对每行代码进行测试




### Training and test

  ```Shell
  # For MARS
  python train.py --root /home/guxinqian/data/ -d mars --arch ap3dres50 --gpu 0,1 --save_dir log-mars-ap3d #
  python test-all.py --root /home/guxinqian/data/ -d mars --arch ap3dres50 --gpu 0 --resume log-mars-ap3d
  
  ```


### Citation

If you use our code in your research or wish to refer to the baseline results, please use the following BibTeX entry.

    @inproceedings{gu2020AP3D,
      title={Appearance-Preserving 3D Convolution for Video-based Person Re-identification},
      author={Gu, Xinqian and Chang, Hong and Ma, Bingpeng and Zhang, Hongkai and Chen, Xilin},
      booktitle={ECCV},
      year={2020},
    }
