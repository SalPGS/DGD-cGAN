#Test model

import os
from os.path import join
import glob
import time
import numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from nets.networks_Dgd import ResUnet1, ResUnet2, PatchDiscriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G1 = ResUnet1().to(device)
G2 = ResUnet2().to(device)
D = PatchDiscriminator(in_c=3, num_filters=64, n_down=3).to(device)


# Download the model weights https://drive.google.com/file/d/11fJ4WrxLCWIF890PvaHp_EuriKwI6QWl/view?usp=share_link
#Load state dic
G1.load_state_dict(torch.load(f'G_dgdgan_epoch_850.pth', map_location=device))
G2.load_state_dict(torch.load(f'G2_dgdgan_epoch_850.pth', map_location=device))
D.load_state_dict(torch.load(f'D_dgdgan_epoch_850.pth', map_location=device))


#Input path
input_path = f"/your/path"
os.chdir(input_path)
img_folder = os.listdir(input_path)

#Output path
output_dir = f"/your/path"

#Mode eval
G1.eval()
D.eval()

#Data transform
transform = transforms.Compose([transforms.Resize((256, 256), transforms.InterpolationMode.BICUBIC)])


## testing 
count = []
d=1

for img in img_folder:
    img_test = transform(Image.open(img))
    img_test = transforms.ToTensor()(img_test)
    input_img = img_test.unsqueeze_(0)
    input_img = input_img.to(device)
    dewatered_img = G1(input_img)
    dewatered_sample = dewatered_img
    image_name = (img.split('.')[-2] +'_%d.jpg'%d)
    file_path = os.path.join(output_dir, image_name)
    save_image(dewatered_sample, file_path, normalize=True)
    d+=1
    
if (len(count) > 1):
    print ("Total imgs: %d" % len(img_folder)) 
