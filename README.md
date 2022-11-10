# DGD-cGAN: A Dual Generator for Image Dewatering and Restoration

This repository contains Pytorch implementation of the paper **"DGD-cGAN: A Dual Generator for Image Dewatering and Restoration"**.
<ul>
  <li> Paper: </li>
  <li> Preprint:</li>
      </ul>
      
      
###### Data preparation  
We used TURBID dataset from Duarte et al.[<a href="http://amandaduarte.com.br/turbid/Turbid_Dataset.pdf" target="_blank">1</a>] available <a href="http://amandaduarte.com.br/turbid/ " target="_blank"> here </a>.
We provide the TURBID images split into four equal sizes. The train data is available for download <a href="https://drive.google.com/file/d/13yxI85JUdsbplM7-Hh8sywIXoom-6hZu/view?usp=sharing" target="_blank"> Ground Truth </a> and <a href="https://drive.google.com/file/d/1XZesr1UCuxnp0gQ3k5tESQd7tkHvCm6t/view?usp=sharing" target="_blank"> Underwater </a> images. Test data is available inside the data folder: [Ground Truth](data/Test_groundtruth.zip) and [Underwater](data/Test_underwater.zip).

###### Representative results

| Underwater Img. | CLAHE | RETINEX |FUnIE-GAN|Water-Net|UGAN|DGD-cGAN|
| ------------- | ------------- | ------------- | ------------- |------------- | ------------- |
| ![](imgs/CLAHE_l2_3deepblue_31_24.jpg) | ![](imgs/CLAHE_l2_3deepblue_31_24.jpg) | ![]imgs/RETINEX_l2_3deepblue_31_24.jpg|![]imgs/FUNIE_GAN_l2_3deepblue_31_24.jpg|![]imgs/WATER_NET_l2_3deepblue_31_24.jpg|![]imgs/UGAN_l2_3deepblue_31_24.jpg|![]imgs/DGD_GAN_l2_3deepblue_31_24.jpg|


DGD-cGAN architecture

![](docs/fig1.png)
   
![](imgs/CLAHE_l2_3deepblue_31_24.jpg)
