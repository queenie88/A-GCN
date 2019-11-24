# A-GCN
## Learning Label Correlations for Multi-Label Image Recognition with Graph Networks

### Requirements

    * Pytorch 0.3.1
    * Python 3.6

### Demo COCO 2014

Please enter the main folder, and run

    python3 demo_coco_gcn.py --data data/coco --image-size 448 --batch-size 32 -e --resume checkpoint/coco/coco_checkpoint.pth.tar

### Our pretrained models

<br>The coco will be update at [GoogleDrive](https://drive.google.com/file/d/1xk-Sh97hpEX7zwc0ZBtnMlHOvN9Id6H5/view?usp=sharing)
<br>The fashion550k will be update at [GoogleDrive](https://drive.google.com/file/d/19cIOOifrf0ww32pLTT6pm_OsmrtEWnEj/view?usp=sharing)


### Citing this repository

If you find our work helpful in your research, please kindly cite our paper:

   Li Q, Peng X, Qiao Y, et al. Learning Category Correlations for Multi-label Image Recognition with Graph Networks[J]. arXiv preprint        arXiv:1909.13005, 2019.
   
 bib:
   
    @article{li2019learning,
    title={Learning Category Correlations for Multi-label Image Recognition with Graph Networks},
    author={Li, Qing and Peng, Xiaojiang and Qiao, Yu and Peng, Qiang},
    journal={arXiv preprint arXiv:1909.13005},
    year={2019}
    }

### Reference

This project is based on https://github.com/Megvii-Nanjing/ML_GCN

