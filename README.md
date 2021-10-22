# SAAT

A Tensorflow implementation of ICCV2021 paper: Improving Robustness of Facial Landmark Detection by Defending against Adversarial Attacks. 
PDF:[SAAT](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhu_Improving_Robustness_of_Facial_Landmark_Detection_by_Defending_Against_Adversarial_ICCV_2021_paper.pdf)

# Installation Instructions
## TensorFlow >= 1.10.1
## Python 2.7
# Different strategies of perturbation generation
<center><img src="https://github.com/zhuccly/SAAT/blob/main/Fig/Attacks.png?raw=true" height=200>
</center>

# Pretrained models and Masked 300W dataset
## Masked 300W: [Download](https://drive.google.com/file/d/1598pCEdSmmubxjCuQ8OdxyG6E833Ybtx/view?usp=sharing)

## The pre-training model and dataset are coming soon.

# Databases
   

    ./databases:
               /ibug:       
                    /image1.jpg     
                     image1.pts       
                     image2.jpg      
                     image2.pts         
               /helen
               /lfpw
               /Masked 300W
            ...  
# bbs

    ./bbs:
         /ibug:
              image1.pts
              image2.pts  
      /helen
      /lfpw
      /Masked 300W
      ...  
# Testing 

Downlod the pre-trained models and put it into ./ckpt/test, then run Evaluation.py

# Bibtex

    @InProceedings{Zhu_2021_ICCV,
    author    = {Zhu, Congcong and Li, Xiaoqiang and Li, Jide and Dai, Songmin},
    title     = {Improving Robustness of Facial Landmark Detection by Defending Against Adversarial Attacks},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {11751-11760}}
   
