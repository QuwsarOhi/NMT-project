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


# Disable tokernizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model = T5().cuda()

# Freezing model layers
if 'freeze_till' in config['model']:
    lst_layer = config['model']['freeze_till']
    for idx, (name, param) in enumerate(model.named_parameters()):
        if idx <= lst_layer:
            param.requires_grad = False
        else:
            break


litmodel = Trainer(model, batch_size=config['dataset']['batch_size'])

trainer = pl.Trainer(**config['trainer'])

#lf_finder = trainer.tuner.lf_find()

trainer.fit(model=litmodel, train_dataloaders=train_data,
            val_dataloaders=val_data)

#trainer.train()