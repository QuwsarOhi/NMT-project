from torch import optim
from model import T5
import lightning.pytorch as pl

class Trainer(pl.LightningModule):
    
    def __init__(self, model:T5, batch_size):
        super().__init__()
        
        self.model = model
        self.batch_size = batch_size
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logit, loss = self.model(x, y)
        self.log("train_loss", loss, batch_size=self.batch_size)
        return loss

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logit, loss = self.model(x, y)
        self.log("val_loss", loss, batch_size=self.batch_size)
        

    def configure_optimizers(self, configs={}):
        optimizer = optim.AdamW(self.parameters(), **configs)
        return optimizer
    
    
