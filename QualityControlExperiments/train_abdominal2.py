import torch
import sys
import numpy as np
from time import time
import math
from torch import Tensor
import yaml
from torch.utils.data import DataLoader
import wandb
import os
import matplotlib.pylab as plt

#sys.path.append("/home/students/studhoene1/imagequality/")

script_dir = os.path.dirname(os.path.abspath(__file__))
shared_dir = os.path.join(script_dir, '..',)
sys.path.append(os.path.abspath(shared_dir))
from simclr.simclr_module2 import SimCLR
from simclr.transforms import SimCLRTrainDataTransform, SimCLREvalDataTransform, SimCLR3SlicesTransform
from simclr.dataset import SimCLR3DDatasetTo2D, SimCLR3DDataset_ForMotion

def fParseConfig(sFile):
    # get config file
    with open(sFile, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def nt_xent_loss(z1, z2, temp, eps=1e-6):
    z = torch.cat([z1, z2], dim=0)

    cosine_sim = torch.nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)

    sim = torch.exp(cosine_sim / temp)
    neg = sim.sum(dim=-1)
    row_sub = Tensor(neg.shape).fill_(math.e ** (1 / temp)).to(neg.device)
    #neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability
    neg = neg - row_sub

    pos_ij = torch.diag(cosine_sim, int(len(z)/2)) # int(len(z)/2 == batch_size
    pos_ji = torch.diag(cosine_sim, -int(len(z)/2))
    pos = torch.cat([pos_ij, pos_ji], dim=0)
    pos = torch.exp(pos / temp)

    loss = -torch.log(pos / (neg + pos)).mean() #try with denom instead neg

    return loss
    
def nt_xent_loss3Batch(z1, z2, zMotion, temp, eps=1e-6):
    z = torch.cat([z1, z2, zMotion], dim=0)

    #cosine_sim = torch.nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)
    cosine_sim = torch.mm(z, torch.transpose(z,0,1))

    sim = torch.exp(cosine_sim / temp)

    neg = torch.cat([sim[:,:int(len(z)/3)], sim[:,-int(len(z)/3):]], dim=1).sum(dim=-1) #only use first and last representation as negative pairs
    neg = torch.cat([neg[:int(len(z)/3)], neg[-int(len(z)/3):]], dim=0)  #only use first and last representation as negative pairs 
    row_sub = Tensor(neg.shape).fill_(math.e ** (1 / temp)).to(neg.device)
    #neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability
    neg = neg - row_sub

    pos_ij = torch.diag(cosine_sim, int(len(z)/3)) # int(len(z)/3 == batch_size 
    pos_ji = torch.diag(cosine_sim, -int(len(z)/3))

    pos_ij = pos_ij[:int(len(z)/3)]  #only positive samples between first two representations
    pos_ji = pos_ji[:int(len(z)/3)]

    pos = torch.cat([pos_ij, pos_ji], dim=0)
    pos = torch.exp(pos / temp)

    loss = -torch.log(pos / (neg + eps)).mean() # +eps instead +pos +pos to avoid negative loss and include all samples form batch to neg, like in original ntxent

    return loss
    
def train(model, dataloader, optimizer, scheduler, device, temperature, epoch):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(dataloader):
        img1, img2, img_motion = batch
        img1, img2, img_motion = img1.to(device), img2.to(device), img_motion.to(device)

        optimizer.zero_grad()
        z1, z2, zMotion = model.train_step(img1, img2, img_motion)

        loss = nt_xent_loss3Batch(z1, z2, zMotion, temperature)
        loss.backward()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
    return running_loss/len(dataloader)



def validate(model, dataloader, device, temperature, epoch):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            img1, img2, img_motion = batch
            img1, img2, img_motion = img1.to(device), img2.to(device), img_motion.to(device)
            
            z1, z2, zMotion = model.train_step(img1, img2, img_motion)
            loss = nt_xent_loss3Batch(z1, z2, zMotion, temperature)
            val_loss += loss.item()
    return val_loss/len(dataloader)
    


if __name__ == "__main__":
    torch.set_printoptions(threshold=99999, edgeitems=1000, linewidth=200)
    print("Start training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    cfg = fParseConfig('/home/students/studhoene1/imagequality/config.yaml')

    wandb.login(key='9506beebc9d4b024ffeb5fba4298098fac09b871')
    wandb.init(project=cfg['WandB_Project'], name=cfg['WandB_Run'])
    wandb.watch_called = False

    model = SimCLR(arch=cfg['arch'])
    model.to(device)
    #checkpoint = torch.load("/home/raecker1/3DSSL/weights/first_wat_crop_02TO1/checkpoints/epoch=372-step=5287275.ckpt")
    #state_dict = checkpoint['state_dict'] 
    #model = SimCLR(arch=cfg['arch'])
    #model.load_state_dict(state_dict)
    #model.to(device)

    #preprocessings = SimCLRTrainDataTransform()
    preprocessing_train = SimCLR3SlicesTransform()
    preprocessing_val = SimCLR3SlicesTransform()
    t_data = time()
    train_dataset = SimCLR3DDataset_ForMotion(cfg['DatasetPath'], cfg['TrainKeysPath'],
                     preprocessing_train, validation=False, small_dataset=True, datatyp=cfg['DataTyp'])
    trainloader = DataLoader(train_dataset, batch_size=cfg['BatchSize'],
                                          shuffle=True, num_workers=0)
    
    val_dataset = SimCLR3DDataset_ForMotion(cfg['DatasetPath'], cfg['ValKeysPath'],
                      preprocessing_val, validation=True, small_dataset=True, datatyp=cfg['DataTyp'])
    valloader = DataLoader(val_dataset, batch_size=cfg['BatchSize'], shuffle=False,
                                         num_workers=0)
    # for batch in trainloader:
    #     print(len(trainloader))
    #     a, b, c = batch
    #     print(a.shape)
    #     plt.imshow(a[0,0,:,:].numpy(), cmap='gray')  # Use a colormap like 'gray' for grayscale images
    #     plt.savefig("AAimage1.png")
    #     plt.show()
    #     plt.imshow(b[0,0,:,:].numpy(), cmap='gray')  # Use a colormap like 'gray' for grayscale images
    #     plt.savefig("AAimage2.png")
    #     plt.show()
    #     plt.imshow(c[0,0,:,:].numpy(), cmap='gray')  # Use a colormap like 'gray' for grayscale images
    #     plt.savefig("AAimage3.png")
    #     plt.show()

    warmup_steps = cfg['WarmupEpochs']*len(trainloader)
    total_steps = cfg['Epochs']*len(trainloader)

    def lr_schedule(step):
        if step <= warmup_steps:
            return step/ max(1, warmup_steps)
        else: 
            progress = (step-warmup_steps) / max(1, total_steps-warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['LearningRate'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    for e in range(cfg['Epochs']):
        t1 = time()
        train_loss = train(model, trainloader, optimizer, scheduler, device, cfg['Temperature'], e)
        val_loss = validate(model, valloader, device, cfg['Temperature'], e)
        print("Train Loss for epoch {}: {:.3f}".format(e+1, train_loss))
        print("Validation Loss for epoch {}: {:.3f}".format(e+1, val_loss))
        print("Time for training epoch {}/{}: {:.2f} Min.".format(e+1, cfg['Epochs'], (time()-t1)/60))

        if (e%2 == 0):
            wandb.log({
                "train loss": train_loss,
                "val loss": val_loss,
                "learning rate": scheduler.get_last_lr()[0],
                "epoch": e+1
            })
        
        if e > 700 and ((e+1) % 50 == 0 or (e+1==1000)):
            model_save_path = os.path.join(cfg['SaveModel'], f"simclr3Slices_changePosNeg_randomMotion_epoch{(e+1)/2}_loss_{train_loss}.pth") #ToDo: Change Model name
            torch.save(model.state_dict(), model_save_path)


    wandb.finish()