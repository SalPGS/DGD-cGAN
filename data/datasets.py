"""
Training DGD-cGAN
"""
#Libraries
import os
import glob
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import itertools



#Datasets
groundtruth = "\ground_truth\" 
path_gt = glob.glob(groundtruth + "\\*.jpg") # Grabbing all the image file names
underwater = "\underwater_images\" 
path_underwater = glob.glob(underwater + "\\*.jpg") # Grabbing all the image file names
airlight = "\airlight_images\"  #324 images with the airlight
path_airlight = glob.glob(airlight + "\\*.jpg") # Grabbing all the image file names
transmission = "\transmission_images\"  #324 images with the transmission
path_transmission = glob.glob(transmission + "\\*.jpg")


# Seed
seed = 128
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#Train and validation datasets
#Choosing 80% for training and 20% testing

#1st dataset is groundtruth dataset which is the gorund truth
np.random.seed(seed)
path_subset_gt = np.random.choice(path_gt, 324, replace=False) 
rand_idxs = np.random.permutation(324)
train_idxs = rand_idxs[:259] 
test_idxs = rand_idxs[259:]
train_path_gt = path_subset_gt[train_idxs]
test_path_gt = path_subset_gt[test_idxs]
print(len(train_path_gt), len(test_path_gt))

#2nd dataset is the underwater dataset 
np.random.seed(seed)
path_subset_underwater = np.random.choice(path_underwater, 324, replace=False)
rand_underwater_idxs = np.random.permutation(324)
train_underwater_idxs = rand_underwater_idxs[:259] 
test_underwater_idxs = rand_underwater_idxs[259:] 
train_path_underwater = path_subset_underwater[train_underwater_idxs]
test_paths_underwater = path_subset_underwater[test_underwater_idxs]
print(len(train_path_underwater),len(test_paths_underwater))

#3rd dataset is the airlight
np.random.seed(seed)
path_subset_airlight = np.random.choice(path_airlight, 324, replace=False) 
rand_airlight_idxs = np.random.permutation(324)
train_airlight_idxs = rand_airlight_idxs[:259] 
test_airlight_idxs = rand_airlight_idxs[259:] 
train_path_airlight = path_subset_airlight[train_airlight_idxs]
test_paths_airlight = path_subset_airlight[test_airlight_idxs]
print(len(train_path_airlight), len(test_paths_airlight))

#4th dataset is the transmission
np.random.seed(seed)
path_subset_transmission = np.random.choice(path_transmission, 324, replace=False) 
rand_transmission_idxs = np.random.permutation(324)
train_transmission_idxs = rand_transmission_idxs[:259] 
test_transmission_idxs = rand_transmission_idxs[259:]
train_path_transmission = path_subset_transmission[train_transmission_idxs]
test_paths_transmission = path_subset_transmission[test_transmission_idxs]
print(len(train_path_transmission), len(test_paths_transmission))

#Loading the data 
#image resize
img_size = 256

class Images_Dataset(Dataset):
    '''
    Function for creating the datasetes, data augmentation, and loading the data
    It has the datasets for training the gan, which are divided into train and test. 
        Values training:
            path_under: underwater images
            path_gt: clear water images which are the ground truth
            path_air: airlight 
            path_trans: transmission 
    '''
    def __init__(self, path_under, path_gt, path_air, path_trans,  split='train'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((img_size, img_size), transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3)
            ])

        elif split == 'test':
            self.transforms = transforms.Resize((img_size, img_size), transforms.InterpolationMode.BICUBIC)

        self.split = split
        self.size = img_size
        self.path_under = path_under
        self.path_gt = path_gt
        self.path_air = path_air
        self.path_trans = path_trans


    
    def __getitem__(self, idx):
    
        img_water = Image.open(self.path_under[idx])
        img_gt = Image.open(self.path_gt[idx])
        img_air = Image.open(self.path_air[idx])
        img_trans = Image.open(self.path_trans[idx])

        img_water = self.transforms(img_water)
        img_with_water = np.array(img_water).astype("float32")
        img_with_water = transforms.ToTensor()(img_with_water)
        
        img_gt = self.transforms(img_gt)
        img_no_water = np.array(img_gt).astype("float32")
        img_no_water = transforms.ToTensor()(img_no_water)
        
        img_air = self.transforms(img_air)
        img_airlight = np.array(img_air).astype("float32")
        img_airlight = transforms.ToTensor()(img_airlight)
        
        img_trans = self.transforms(img_trans)
        img_transmission = np.array(img_trans).astype("float32")
        img_transmission = transforms.ToTensor()(img_transmission)
        
        
    

        Uw= img_with_water /255.
        Gt = img_no_water /255.
        A = img_airlight /255.
        T = img_transmission /255.
        AT = torch.cat([A,T], dim=0)
        

        return {'Uw': Uw, 'Gt': Gt,  'A': A, 'T': T, 'AT': AT}

  
    
    def __len__(self):
    
        return len(self.path_gt)


def load_data(batch_size=5, n_workers=0, pin_memory=True, **kwargs):
    dataset_images = Images_Dataset(**kwargs)
    dataloader= DataLoader(dataset_images, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader
