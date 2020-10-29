# A-GCN
## Learning Label Correlations for Multi-Label Image Recognition with Graph Networks

### Requirements

    * Pytorch 0.3.1
    * Python 3.6

### Demo COCO 2014

Please enter the main folder, and run

    python3 demo_coco__adapt.py --data data/coco --image-size 448 --batch-size 32 -e --resume checkpoint/coco/coco_checkpoint.pth.tar

### Our pretrained models

<br>The pre-trained model on coco at [GoogleDrive](https://drive.google.com/file/d/1xk-Sh97hpEX7zwc0ZBtnMlHOvN9Id6H5/view?usp=sharing)
<br>The pre-trained model on fashion550k at [GoogleDrive](https://drive.google.com/file/d/19cIOOifrf0ww32pLTT6pm_OsmrtEWnEj/view?usp=sharing)


### Citing this repository

If you find our work helpful in your research, please kindly cite our paper:

Li Q, Peng X, Qiao Y, et al. Learning label correlations for multi-label image recognition with graph networks[J]. Pattern Recognition Letters, 2020, 138: 378-384.
   
 bib:
   
 @article{li2020learning,
   title={Learning label correlations for multi-label image recognition with graph networks},
   author={Li, Qing and Peng, Xiaojiang and Qiao, Yu and Peng, Qiang},
   journal={Pattern Recognition Letters},
   volume={138},
   pages={378--384},
   year={2020},
   publisher={Elsevier}
 }
### Reference

This project is based on https://github.com/Megvii-Nanjing/ML_GCN

