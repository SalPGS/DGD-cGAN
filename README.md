# DGD-cGAN: A Dual Generator for Image Dewatering and Restoration

This repository contains Pytorch implementation of the paper **"DGD-cGAN: A Dual Generator for Image Dewatering and Restoration"**.
<ul>
  <li> Paper: </li>
  <li> Preprint:</li>
      </ul>
<br>

###### Abstract
Underwater images are usually covered with a blue-greenish colour cast, making them distorted,
blurry or low in contrast. This phenomenon occurs due to the light attenuation given by the scattering
and absorption in the water column. In this paper, we present an image enhancement approach for
dewatering which employs a conditional generative adversarial network (cGAN) with two generators.
Our Dual Generator Dewatering cGAN (DGD-cGAN) removes the haze and colour cast induced
by the water column and restores the true colours of underwater scenes whereby the effects of
various attenuation and scattering phenomena that occur in underwater images are tackled by the two
generators. The first generator takes at input the underwater image and predicts the dewatered scene,
while the second generator learns the underwater image formation process by implementing a custom
loss function based upon the transmission and the veiling light components of the image formation
model. Our experiments show that DGD-cGAN consistently delivers a margin of improvement as
compared with the state-of-the-art methods on several widely available datasets.
      
<br>

###### Data preparation  
We used TURBID dataset from Duarte et al.[<a href="http://amandaduarte.com.br/turbid/Turbid_Dataset.pdf" target="_blank">1</a>] available <a href="http://amandaduarte.com.br/turbid/ " target="_blank"> here </a>.
We provide the TURBID images split into four equal sizes. The data is available for download <a href="https://drive.google.com/file/d/13yxI85JUdsbplM7-Hh8sywIXoom-6hZu/view?usp=sharing" target="_blank"> Ground Truth </a>, <a href="https://drive.google.com/file/d/1XZesr1UCuxnp0gQ3k5tESQd7tkHvCm6t/view?usp=sharing" target="_blank"> Underwater </a>, <a href="https://drive.google.com/file/d/1Nf9RJD5GhFBNhNcT0BGWBy7hB0cBoRFE/view?usp=sharing" target="_blank"> Transmission </a> and <a href="https://drive.google.com/file/d/1HifH7pv9NRrzwrwK9cv_axViOH-n1THJ/view?usp=sharing" target="_blank"> Scattered light </a> images. Test data is available inside the data folder: [Ground Truth](data/Test_groundtruth.zip) and [Underwater](data/Test_underwater.zip).

<br>

###### DGD-cGAN architecture

<img align="centre" src="https://github.com/SalPGS/DGD-cGAN/blob/edc60bc89f7738724a6907a689f28517ddeb8b3b/docs/fig1.png">

<br>

###### Representative results

We compare DGD-cGAN, with Contrast Limited Adaptive Histogram Equalisation (CLAHE), Retinex, FUnIE-GAN, Water-Net and UGAN methods.

| Image Input | Ground truth | CLAHE | RETINEX | FUnIE-GAN | Water-Net | UGAN | DGD-cGAN | 
|     :---:      |     :---:      |     :---:      |     :---:      |     :---:     |     :---:      |     :---:      |     :---:      |          
|<img class="imgs-1" src="https://github.com/SalPGS/DGD-cGAN/blob/8ededbb74900ddf1af11a01dd951696dd23b5ac5/docs/imgs/UNDERWATER_l2_3deepblue_31_24.jpg">|<img src="https://github.com/SalPGS/DGD-cGAN/blob/8ededbb74900ddf1af11a01dd951696dd23b5ac5/docs/imgs/GROUND_TRUTH_l2_3deepblue_31_24.jpg">|<img src="https://github.com/SalPGS/DGD-cGAN/blob/8ededbb74900ddf1af11a01dd951696dd23b5ac5/docs/imgs/CLAHE_l2_3deepblue_31_24.jpg">|<img src="https://github.com/SalPGS/DGD-cGAN/blob/8ededbb74900ddf1af11a01dd951696dd23b5ac5/docs/imgs/RETINEX_l2_3deepblue_31_24.jpg">|<img src="https://github.com/SalPGS/DGD-cGAN/blob/8ededbb74900ddf1af11a01dd951696dd23b5ac5/docs/imgs/FUNIE_GAN_l2_3deepblue_31_24.jpg">|<img src="https://github.com/SalPGS/DGD-cGAN/blob/8ededbb74900ddf1af11a01dd951696dd23b5ac5/docs/imgs/WATER_NET_l2_3deepblue_31_24.jpg">|<img src="https://github.com/SalPGS/DGD-cGAN/blob/8ededbb74900ddf1af11a01dd951696dd23b5ac5/docs/imgs/UGAN_l2_3deepblue_31_24.jpg">|<img src="https://github.com/SalPGS/DGD-cGAN/blob/8ededbb74900ddf1af11a01dd951696dd23b5ac5/docs/imgs/DGD_GAN_l2_3deepblue_31_24.jpg">|
|<img class="imgs-1" src="https://github.com/SalPGS/DGD-cGAN/blob/2c956e82e1b89f6a1bb020c5ed792ba099d6e8cf/docs/imgs/UNDERWATER_l1_3_8_9.jpg">|<img src="https://github.com/SalPGS/DGD-cGAN/blob/2c956e82e1b89f6a1bb020c5ed792ba099d6e8cf/docs/imgs/GROUND_TRUTH_l1_3_8_9.jpg">|<img src="https://github.com/SalPGS/DGD-cGAN/blob/2c956e82e1b89f6a1bb020c5ed792ba099d6e8cf/docs/imgs/CLAHE_l1_3_8_9.jpg">|<img src="https://github.com/SalPGS/DGD-cGAN/blob/ccfb9548c9716670e9481fe34d699cfae5e44313/docs/imgs/RETINEX_l1_3_8_9.jpg">|<img src="https://github.com/SalPGS/DGD-cGAN/blob/2c956e82e1b89f6a1bb020c5ed792ba099d6e8cf/docs/imgs/FUNE_GAN_l1_3_8_9.jpg">|<img src="https://github.com/SalPGS/DGD-cGAN/blob/2c956e82e1b89f6a1bb020c5ed792ba099d6e8cf/docs/imgs/WATER_NET_l1_3_8_9.jpg">|<img src="https://github.com/SalPGS/DGD-cGAN/blob/2c956e82e1b89f6a1bb020c5ed792ba099d6e8cf/docs/imgs/UGAN_l1_3_8_9.jpg">|<img src="https://github.com/SalPGS/DGD-cGAN/blob/2c956e82e1b89f6a1bb020c5ed792ba099d6e8cf/docs/imgs/DGD_cGAN_l1_3_8_9.jpg">|
