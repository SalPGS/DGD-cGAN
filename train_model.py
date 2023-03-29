"""
Training DGD-cGAN
"""
#Libraries
import os
import glob
import numpy as np
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import itertools
import random

#local libraries
from data.datasets import load_data
from nets.networks_Dgd import ResUnet1, ResUnet2, PatchDiscriminator
from nets.losses import GANLoss, init_weights
from nets.utils import loss_results, create_loss_meters, update_losses


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Datasets
train_data = load_data( path_under= train_path_underwater, path_gt=train_path_gt, path_air=train_path_airlight, path_trans=train_path_transmission, split='train')
test_data = load_data(path_under= test_paths_underwater, path_gt=test_path_gt, path_air=test_paths_airlight, path_trans=test_paths_transmission, split='test')



class DGD_cGAN(nn.Module):
    def __init__(self, device=device, lr_Gens= 0.0002 , lr_D= 0.0002, 
                 beta1=0.5, beta2=0.999, lamb_G1=100.0, lamb_G2=0.5):
        super().__init__()
        
        self.lamb_L1_G1 = lamb_G1
        self.lamb_L2_G2 = lamb_G2
        self.G1 = init_weights(ResUnet1(), device)
        self.G2 = init_weights(ResUnet2(), device)
        self.D = init_weights(PatchDiscriminator(in_c=3, num_filters=64, n_down=3), device)
        self.Adv_Loss = GANLoss().to(device)
        self.L1_G1 = nn.L1Loss()
        self.L2_G2 = nn.L1Loss()
        self.opt_Gens = optim.Adam(itertools.chain(self.G1.parameters(), self.G2.parameters()), lr=lr_Gens, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.D.parameters(), lr=lr_D, betas=(beta1, beta2))
        
    
    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad
        
    def set_input(self, data):
        self.Uw = data['Uw'].to(device)
        self.Gt = data['Gt'].to(device)
        self.AT = data['AT'].to(device)

        
    def forward(self):
        self.fake_dewater = self.G1(self.Uw)
        self.fake_G2 = self.G2(self.AT)

    
    def backward_D(self):
        fake_img= self.fake_dewater
        fake_pred = self.D(fake_img.detach())
        self.D_fake_Loss = self.Adv_Loss(fake_pred, False)
        real_img = self.Gt
        real_pred = self.D(real_img)
        self.D_real_Loss = self.Adv_Loss(real_pred, True)
        self.D_Loss = (self.D_fake_Loss + self.D_real_Loss) * 0.5
        self.D_Loss.backward()
    
    def backward_G(self):
        fake_img= self.fake_dewater
        fake_pred = self.D(fake_img)
        self.G_Loss = self.Adv_Loss(fake_pred, True)
        self.G2_A = self.fake_G2[:, 0:3]
        self.G2_T = self.fake_G2[:, 3:6]

        self.fake_N = (self.fake_dewater * self.G2_T) + self.G2_A
        
        self.G1_L1_Loss = self.L1_G1(self.fake_dewater, self.Gt) * self.lamb_L1_G1
        self.G2_L2_Loss = self.L2_G2(self.fake_N, self.Uw) * self.lamb_L2_G2 

        self.loss_Gens = self.G_Loss + self.G1_L1_Loss + self.G2_L2_Loss 
        self.loss_Gens.backward()
    
    def optimizer(self):
        self.forward()
        self.D.train()
        self.set_requires_grad(self.D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()
        
        self.G1.train()
        self.G2.train()
        self.set_requires_grad(self.D, False)
        self.opt_Gens.zero_grad()
        self.backward_G()
        self.opt_Gens.step()
        
        
epochs=850
def train_model(model, train_data, epochs, display_every=16):
    data = next(iter(test_data))
    G1L1_Loss = []
    G2L2_Loss = []
    for e in range(epochs):
        loss_meter_dict = create_loss_meters()
        i = 0  

        for data in tqdm(train_data):
            model.set_input(data) 
            model.optimizer()
            update_losses(model, loss_meter_dict, count=data['Uw'].size(0)) 
            G1L1_Loss.append(loss_meter_dict["G1_L1_Loss"].avg)
            G2L2_Loss.append(loss_meter_dict["G2_L2_Loss"].avg)
            

            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_data)}")
                loss_results(loss_meter_dict) 
                
        

model = DGD_cGAN()
train_model(model, train_data, epochs)
