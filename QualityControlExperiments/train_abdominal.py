import torchvision
import torch

import numpy as np

import torchio as tio
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning import seed_everything

import torchio as tio

from selfsupervised2d.simclr.simclr_module import SimCLR
from selfsupervised2d.simclr.transforms import SimCLRTrainDataTransform, SimCLREvalDataTransform
from selfsupervised2d.simclr.dataset import SimCLR2DDataset

seed_everything(0) 
print("START")
blur_params = (0, 2)
noise_params = {"std":(0, 0.1), "p":0.3}
gamma_params = (-0.5, 0.5)
random_crop_params = (0.2, 1)

gpus = 1
batch_size = 4
preprocessings = SimCLRTrainDataTransform()

train_dataset = SimCLR2DDataset('/mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/raw/', "/home/raecker1/3DSSL/selfsupervised2d/ukb_abdominal_train_keys.npy",
                     preprocessings, False)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

val_dataset = SimCLR2DDataset('/mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/raw/', "/home/raecker1/3DSSL/selfsupervised2d/ukb_abdominal_val_keys.npy",
                     SimCLREvalDataTransform())

valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                        num_workers=0)

simclr = SimCLR(gpus, 100000, batch_size)
trainer = pl.Trainer(gpus=gpus, default_root_dir=".", max_epochs=1000, num_sanity_val_steps=0, strategy="ddp")
trainer.fit(simclr, train_dataloaders=trainloader, val_dataloaders=valloader)
