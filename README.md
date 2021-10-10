# SAAT

A Tensorflow implementation of ICCV2021 paper: Improving Robustness of Facial Landmark Detection by Defending against Adversarial Attacks. 
PDF:https://openaccess.thecvf.com/content/ICCV2021/papers/Zhu_Improving_Robustness_of_Facial_Landmark_Detection_by_Defending_Against_Adversarial_ICCV_2021_paper.pdf

# TensorFlow >= 1.10.1

# Pretrained models and Masked 300W dataset

The pre-training model and dataset are coming soon.
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

