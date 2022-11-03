import numpy as np
import matplotlib.pyplot as plt
import torch


#Loss meter
class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count

def create_loss_meters():
    D_fake_Loss = AverageMeter()
    D_real_Loss = AverageMeter()
    D_Loss = AverageMeter()
    G_Loss = AverageMeter()
    G1_L1_Loss = AverageMeter()
    loss_Gens = AverageMeter()
    G2_L2_Loss  = AverageMeter()

    
    
    return {'D_fake_Loss': D_fake_Loss,
            'D_real_Loss': D_real_Loss,
            'D_Loss': D_Loss,
            'G_Loss': G_Loss,
            'G1_L1_Loss': G1_L1_Loss,
            'loss_Gens': loss_Gens,
            'G2_L2_Loss':G2_L2_Loss
            }



def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count) 


        
def loss_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")
