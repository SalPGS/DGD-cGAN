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
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()
    loss_G_L2  = AverageMeter()

    
    
    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G,
            'loss_G_L2':loss_G_L2
            }



def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count) 

# Display images    
def grid_output(model, data):
    model.net_G.eval()
    with torch.no_grad():
        model.set_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.Gt
    Uw = model.Uw.cpu().numpy()
    Uw = np.transpose(Uw,(0,2,3,1))

    fake_imgs=fake_color.cpu().numpy()
    fake_imgs = np.transpose(fake_imgs,(0,2,3,1))

    real_imgs=real_color.cpu().numpy()
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
