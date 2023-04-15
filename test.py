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

def main():
    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    seed_everything(SEED)

    # loading config file
    with open("config.json", "r") as f:
        config = json.load(f)

    # Disable tokernizer parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = T5().cuda()

    # Loading weights
    print("Model weight", config["weight"])
    model.load_state_dict(torch.load(config["weight"]))

    train_data, val_data, test_data = get_dataset(**config['dataset'])

    litmodel = Trainer(model, batch_size=config['dataset']['batch_size'],
                       optim_args=config['optim_args'])

    # Adding progress bar by default
    config['trainer']['enable_progress_bar'] = True
    trainer = pl.Trainer(**config['trainer'])

    #lf_finder = trainer.tuner.lf_find()
    
    #print("\nValidation started for : "+Dataset(config_name['ids']))
    print("\n",type(val_data),"\n",val_data)
    #trainer.validate(model=litmodel, dataloaders=val_data)
    
    #print("\nValidation completed for : "+config['dataset']['ids'])
    
    #print("\nTesting started for : "+config['dataset']['ids'])
    
    #trainer.test(model=litmodel, dataloaders=test_data)

    #print("\nTesting completed for : "+config['dataset']['ids'])

def sanity_test():
    print("Sanity Testing")

    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    seed_everything(SEED)

    # loading config file
    with open("config.json", "r") as f:
        config = json.load(f)

    # Disable tokernizer parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = T5().cuda()
    
    # Loading weights
    print("Model weight", config["weight"])
    model.load_state_dict(torch.load(config["weight"]))

    inputs = ["Good Morning, How are you?"]
    targets = ["Buongiorno, come stai?"]

    logits, loss = self.forward(input=input)
    outputs = self.tokenizer.batch_decode(logits, skip_special_tokens=True)

    for (inp, out), tar in zip(zip(inputs, outputs), targets):
        print(f"Input: \n{inp}\n\nOutput: \n{out}\n\nTarget: \n{tar}\n\n")
    

if __name__ == "__main__":
    
    main()

    #sanity_test()