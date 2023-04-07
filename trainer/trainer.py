from torch import optim
from model import T5
import lightning.pytorch as pl
from torchtext.data.metrics import bleu_score


class Trainer(pl.LightningModule):
    
    def __init__(self, model:T5, batch_size, optim_args):
        super().__init__()
        
        self.model = model
        self.batch_size = batch_size
        self.optim_args = optim_args
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logit, loss = self.model(x, y)
        self.log("train_loss", loss, batch_size=self.batch_size)
        return loss

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logit, loss = self.model(x, y)
        out = self.model.predict(x)
        
        pred, ground = [], []
        
        for p, g in zip(out, y):
            pred.append(p.split())
            ground.append([g.split()])
        
        self.log_dict({"val_loss": loss, 
                       "val_bleu": bleu_score(pred, ground)},
                      batch_size=self.batch_size)
    
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logit, loss = self.model(x, y)
        out = self.model.predict(x)
        
        pred, ground = [], []
        
        for p, g in zip(out, y):
            pred.append(p.split())
            ground.append([g.split()])
        
        self.log_dict({"test_loss": loss, 
                       "test_bleu": bleu_score(pred, ground)},
                      batch_size=self.batch_size)
        

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), **self.optim_args)
        return optimizer
    
    
