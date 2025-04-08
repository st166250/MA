import torchvision
import torch
import sys
import numpy as np
import os
import torchio as tio
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning import seed_everything

import torchio as tio
from torch.utils.data import DataLoader, SubsetRandomSampler
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.model_summary import summarize

from pytorch_lightning.callbacks import LearningRateMonitor


sys.path.append("/home/students/studhoene1/imagequality/")
from simclr.simclr_module import SimCLR
from simclr.transforms import SimCLRTrainDataTransform, SimCLREvalDataTransform
from simclr.dataset import SimCLR2DDataset, SimCLR3DDataset, SimCLR3DDatasetTo2D

def main():

    seed_everything(0) 
    print("START")

    gpus = 1
    batch_size = 128

    blur_params = (0, 2)
    noise_params = {"std":(0, 0.1), "p":0.3}
    gamma_params = (-0.5, 0.5)
    random_crop_params = (0.2, 1)

    preprocessings = SimCLRTrainDataTransform()

    train_dataset = SimCLR3DDatasetTo2D('/mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/raw/', "/home/students/studhoene1/imagequality/ukb_abdominal_train_keys.npy",
                     preprocessings, False, small_dataset=True)
    print(f"Dataset size: {len(train_dataset)}")

    trainloader = DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
    val_dataset = SimCLR3DDatasetTo2D('/mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/raw/', "/home/students/studhoene1/imagequality/ukb_abdominal_val_keys.npy",
                      SimCLREvalDataTransform(), validation=True, small_dataset=True)

    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=0)

    checkpoint_callback = ModelCheckpoint(
        monitor = 'train_loss',
        dirpath = './checkpoints',
        filename = 'from_start_pretrained_{epoch:02d}_{train_loss:.2f}',
        save_top_k = -1,
        mode = 'min',
        save_last = True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')  # Logs at each step

    model = SimCLR.load_from_checkpoint("/home/raecker1/3DSSL/weights/first_wat_crop_02TO1/checkpoints/epoch=372-step=5287275.ckpt")
    #simclr = SimCLR(gpus, 6144, batch_size)
    trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir=".", max_epochs=1000, callbacks=[checkpoint_callback, lr_monitor], num_sanity_val_steps=0)#, strategy="ddp")
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)

if __name__ == "__main__":
    main()