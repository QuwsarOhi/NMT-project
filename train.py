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

# Data-split for Train:Validation:Test = 80:10:10
train_data, val_data, test_data = get_dataset(**config['dataset'])


# Disable tokernizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model = T5().cuda()
#model.to('cuda')

# Freezing model layers
if 'freeze_till' in config['model']:
    lst_layer = config['model']['freeze_till']
    for idx, (name, param) in enumerate(model.named_parameters()):
        if idx <= lst_layer:
            param.requires_grad = False
        else:
            break


litmodel = Trainer(model, batch_size=config['dataset']['batch_size'],
                   optim_args=config['optim_args'])


checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                      save_top_k=1,
                                      mode='min',
                                      dirpath=config['trainer']['default_root_dir'])

early_stopping = EarlyStopping(monitor='val_loss', 
                               patience=10, 
                               verbose=True,
                               mode='min')

trainer = pl.Trainer(**config['trainer'], 
                     callbacks=[checkpoint_callback, early_stopping]
                    )

#lf_finder = trainer.tuner.lf_find()

trainer.fit(model=litmodel, train_dataloaders=train_data,
            val_dataloaders=val_data, **config['fit'])
