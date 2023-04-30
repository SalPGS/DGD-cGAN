"""
 > Network modules of DGD-cGAN model
   * Paper: 
"""
#Libraries
import torch
from torch import nn
#First GENERATOR G1

class batchnorm_1(nn.Module):
    def __init__(self, in_c):
        super().__init__()       
        self.bn1 = nn.BatchNorm2d(in_c)
        self.relu1 = nn.LeakyReLU(0.2, True)
        
    def forward(self, inputs):
        x = self.bn1(inputs)
        x = self.relu1(x)
        return x
    

class residual_unit1(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.batch1 = batchnorm_1(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.batch2 = batchnorm_1(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1)
        self.s = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=stride)

    def forward(self, inputs):
        x = self.batch1(inputs)
        x = self.conv1(x)
        x = self.batch2(x)
        x = self.conv2(x)
        s = self.s(inputs)
        skip = x + s
        return skip

class decoder1(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.uprelu = nn.ReLU(True)
        self.res = residual_unit1(in_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = self.uprelu(x)
        x = torch.cat([x, skip], axis=1)
        x = self.res(x)
        return x

    
class ResUnet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.batch1 = batchnorm_1(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(3, 64, kernel_size=1, padding=0)
        self.res2 = residual_unit1(64, 128, stride=2)
        self.res3 = residual_unit1(128, 256, stride=2)
        self.res4 = residual_unit1(256, 512, stride=2)
        self.dec1 = decoder1(512, 256)
        self.dec2 = decoder1(256, 128)
        self.dec3 = decoder1(128, 64)
        self.output = nn.Conv2d(64, 3, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.batch1(x)
        x = self.conv2(x)
        s = self.conv3(inputs)
        skip1 = x + s
        skip2 = self.res2(skip1)
        skip3 = self.res3(skip2)
        b = self.res4(skip3)
        dec1 = self.dec1(b, skip3)
        dec2 = self.dec2(dec1, skip2)
        dec3 = self.dec3(dec2, skip1)
        output = self.output(dec3)
        output = self.sigmoid(output)
        return output

#2nd GENERATOR G2

#2nd GENERATOR G2

class batchnorm_2(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.bn2 = nn.BatchNorm2d(in_c)
        self.relu2 = nn.LeakyReLU(0.2, True)
        
    def forward(self, inputs):
        x = self.bn2(inputs)
        x = self.relu2(x)
        return x
    

class residual_unit2(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.batch1 = batchnorm_2(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.batch2 = batchnorm_2(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1)
        self.s = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=stride)

    def forward(self, inputs):
        x = self.batch1(inputs)
        x = self.conv1(x)
        x = self.batch2(x)
        x = self.conv2(x)
        s = self.s(inputs)
        skip = x + s
        return skip

class decoder2(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.uprelu = nn.ReLU(True)
        self.res = residual_unit2(in_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = self.uprelu(x)
        x = torch.cat([x, skip], axis=1)
        x = self.res(x)
        return x

    
class ResUnet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.batch1 = batchnorm_2(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(6, 64, kernel_size=1, padding=0)
        self.res2 = residual_unit2(64, 128, stride=2)
        self.res3 = residual_unit2(128, 256, stride=2)
        self.res4 = residual_unit2(256, 512, stride=2)
        self.dec1 = decoder2(512, 256)
        self.dec2 = decoder2(256, 128)
        self.dec3 = decoder2(128, 64)
        self.output = nn.Conv2d(64, 6, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.batch1(x)
        x = self.conv2(x)
        s = self.conv3(inputs)
        skip1 = x + s
        skip2 = self.res2(skip1)
        skip3 = self.res3(skip2)
        b = self.res4(skip3)
        dec1 = self.dec1(b, skip3)
        dec2 = self.dec2(dec1, skip2)
        dec3 = self.dec3(dec2, skip1)
        output = self.output(dec3)
        output = self.sigmoid(output)
        return output

#Patch Discriminator

class PatchDiscriminator(nn.Module):
    def __init__(self, in_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(in_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down-1) else 2) 
                          for i in range(n_down)] 
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)]                                                                                              
        self.model = nn.Sequential(*model)                                                   
        
    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True): 
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]           
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.ReLU6()]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
