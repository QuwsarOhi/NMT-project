import torch
import json
import numpy as np
import os, shutil, sys
import lightning.pytorch as pl
from lightning.pytorch import seed_everything

from trainer.trainer import Trainer
from dataloader.dataloader import get_dataset
from model.T5 import T5


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)
seed_everything(SEED)


# loading config file
with open("config.json", "r") as f:
    config = json.load(f)

train_data, val_data, test_data = get_dataset(**config['dataset'])

model = T5()
litmodel = Trainer(model, batch_size=config['dataset']['batch_size'])

print("DONE")

trainer = pl.Trainer(limit_train_batches=100, max_epochs=1,
                     deterministic=True,
                     accelerator="cpu", 
                     #auto_lr_find=True
                     )

#lf_finder = trainer.tuner.lf_find()

trainer.fit(model=litmodel, train_dataloaders=train_data,
            val_dataloaders=val_data)

#trainer.train()