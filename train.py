import torch
import json
from trainer.trainer import Trainer
from dataloader.dataloader import get_dataset
from T5 import T5
import os, shutil, sys

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


# loading config file
with open("config.json", "r") as f:
    config = json.load(f)

train_data, val_data, test_data = get_dataset(**config['dataset'])

#model = T5(
#
#)

#trainer = Trainer(
#
#)

#trainer.train()