import torch
import json
import numpy as np
import os, shutil, sys
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

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

# Loading weights
model.load_state_dict(torch.load('/home/btlab/Ohi/NMT-project/T5-v2.pth'))

litmodel = Trainer(model, batch_size=config['dataset']['batch_size'],
                   optim_args=config['optim_args'])

trainer = pl.Trainer(**config['trainer'])

#lf_finder = trainer.tuner.lf_find()

trainer.test(model=litmodel, dataloaders=test_data)

trainer.validate(model=litmodel, dataloaders=val_data)