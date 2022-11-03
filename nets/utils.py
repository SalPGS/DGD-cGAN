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



# Display images    
def grid_output(model, data):
    model.G1.eval()
    with torch.no_grad():
        model.set_input(data)
        model.forward()
    model.G1.train()
    fake_dewater = model.fake_dewater.detach()
    groun_truth = model.Gt
    Uw = model.Uw.cpu().numpy()
    Uw = np.transpose(Uw,(0,2,3,1))

    fake_imgs= fake_dewater.cpu().numpy()
    fake_imgs = np.transpose(fake_imgs,(0,2,3,1))

    real_imgs= groun_truth.cpu().numpy()
    real_imgs = np.transpose(real_imgs,(0,2,3,1))

    fig = plt.figure(figsize=(15, 8))

    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(Uw[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()

        
def loss_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")
