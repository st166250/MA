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
from torchsummary import summary
import matplotlib.pylab as plt
from datetime import datetime, timedelta

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch_optimizer as optim
#sys.path.append("/home/students/studhoene1/imagequality/")

script_dir = os.path.dirname(os.path.abspath(__file__))
shared_dir = os.path.join(script_dir, '..',)
sys.path.append(os.path.abspath(shared_dir))
from simclr.simclr_module2 import SimCLR
from simclr.transforms import FineTuneTransform
from simclr.dataset import NakoIQADataset_SimCLR, SimCLR3DDataset_ForMotion_V2

from backbone.ssl_head import SSLHead

def fParseConfig(sFile):
    # get config file
    with open(sFile, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

def nt_xent_loss3Batch(z1, z2, zMotion, temp, eps=1e-6):

    z1_dist = z1
    z2_dist = z2
    zMotion_dist = zMotion
    
    z = torch.cat([z1, z2, zMotion], dim=0)
    z_dist = torch.cat([z1_dist, z2_dist, zMotion_dist], dim=0)

    #cosine_sim = torch.nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)
    #cosine_sim = torch.mm(z, torch.transpose(z,0,1))
    cosine_sim = torch.mm(z, z_dist.t().contiguous())

    sim = torch.exp(cosine_sim / temp)

    neg = torch.cat([sim[:,:int(len(z_dist)/3)], sim[:,-int(len(z_dist)/3):]], dim=1).sum(dim=-1) #only use first and last representation as negative pairs
    neg = torch.cat([neg[:int(len(z)/3)], neg[-int(len(z)/3):]], dim=0)  #only use first and last representation as negative pairs 
    row_sub = Tensor(neg.shape).fill_(math.e ** (1 / temp)).to(neg.device)
    #neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability
    neg = neg - row_sub

    pos_ij = torch.diag(cosine_sim, int(len(z_dist)/3)) # int(len(z)/3 == batch_size 
    pos_ji = torch.diag(cosine_sim, -int(len(z)/3))

    pos_ij = pos_ij[:int(len(z)/3)]  #only positive samples between first two representations
    pos_ji = pos_ji[:int(len(z)/3)]

    pos = torch.cat([pos_ij, pos_ji], dim=0)
    pos = torch.exp(pos / temp)

    loss = -torch.log(pos / (neg + pos)).mean() # +eps instead +pos +pos to avoid negative loss and include all samples form batch to neg, like in original ntxent

    return loss.contiguous()

    
def train(model, dataloader, optimizer, scheduler, device, temperature, epoch):
    model.encoder.eval()
    model.projection.train()
    running_loss = 0.0

    for i, batch in enumerate(dataloader):
        img_hq1, img_hq2, img_lq = batch
        print(img_hq1.shape)
        img_hq1, img_hq2 , img_lq = img_hq1.to(device), img_hq2.to(device), img_lq.to(device)

        optimizer.zero_grad()
        z1, z2, z3 = model.finetune_step(img_hq1, img_hq2, img_lq)

        loss = nt_xent_loss3Batch(z1, z2, z3, temperature)
        loss.backward()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss

    return running_loss.item() / len(dataloader)



def validate(model, dataloader, device, temperature, epoch):
    model.projection.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            img_hq1, img_hq2, img_lq = batch
            img_hq1, img_hq2 , img_lq = img_hq1.to(device), img_hq2.to(device), img_lq.to(device)
            
            z1, z2, z3 = model.finetune_step(img_hq1, img_hq2, img_lq)
            loss = nt_xent_loss3Batch(z1, z2, z3, temperature)
            val_loss += loss


    return val_loss.item() / len(dataloader)
    


def main():
    torch.set_printoptions(threshold=99999, edgeitems=1000, linewidth=200)
    print("Start training")


    device = torch.device("cuda:0")
    print("Using device: {}".format(device))

    cfg = fParseConfig('/home/students/studhoene1/imagequality/config_finetune.yaml')

    wandb.login(key='9506beebc9d4b024ffeb5fba4298098fac09b871')
    wandb.init(project=cfg['WandB_Project'], name=cfg['WandB_Run'])
    wandb.watch_called = False

  
    model = SimCLR(arch=cfg['arch'])
    checkpoint = torch.load('/home/students/studhoene1/imagequality/QualityControlExperiments/checkpoints_4GPU/simclr3Slices5000.0_loss_5.590002059936523.pth')
    new_state_dict = {}
    for key, value in checkpoint.items():
        new_key = key.replace("module.", "")  # Remove 'module.' from keys
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.to(device)

    #preprocessings = SimCLRTrainDataTransform()
    preprocessing_finetuning_train = FineTuneTransform()
    preprocessing_finetuning_val = FineTuneTransform()

    t_data = time()
    train_dataset = NakoIQADataset_SimCLR("/mnt/qdata/rawdata/NAKO_IQA/NAKO_IQA_nifti/", preprocessing_finetuning_train, validation=False)
    trainloader = DataLoader(train_dataset, batch_size=cfg['BatchSize'], num_workers=0, pin_memory=True)
    
    val_dataset = NakoIQADataset_SimCLR("/mnt/qdata/rawdata/NAKO_IQA/NAKO_IQA_nifti/", preprocessing_finetuning_val, validation=True)
    valloader = DataLoader(val_dataset, batch_size=cfg['BatchSize'], num_workers=0, pin_memory=True)

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
        
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg['LearningRate'])
    #optimizer = torch.optim.Adam(model.parameters(), lr=cfg['LearningRate'])
    #optimizer = optim.LARS(model.parameters(), lr=cfg['LearningRate'], weight_decay=1e-6, momentum=0.9)
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
        
        if  ((e+1) % 500 == 0 or (e+1==5000)):
            model_save_path = os.path.join(cfg['SaveModel'], f"simclr3Slices{(e+1)}_loss_{val_loss}.pth") #ToDo: Change Model name
            torch.save(model.state_dict(), model_save_path)

    wandb.finish()


if __name__ == "__main__":
    main()