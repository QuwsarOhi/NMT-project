from torch.nn import Module
from torch.utils.data import DataLoader
import torch, time
import numpy as np
import pickle, os

class Trainer:

    def __init__(self, 
                 model:Module,                          # the model to train
                 device:str,                            # device to train on
                 epoch:int,                             # training epoch
                 train_data:DataLoader,                 # training dataset
                 val_data:DataLoader=None,              # validation dataset
                 prev_checkpoint:str='',                # start from the previous checkpoint (path)
                 prev_weight:str='',                    # load previous weight (path)
                 save_best_weight:bool=False,           # save the model weight whenever it performs best 
                 monitor:str='train_loss',              # which metric to monitor
                 mode:str='min',                        # which quantity is the best: minimum or maximum
                 checkpoint_freq=0,                     # checkpoint epoch duration
                 filepath='./',                         # where to save the checkpoint and weight
                ):
        

        assert mode in ['min', 'max'], "mode can be either 'min' or 'max'"

        self.model                  = model
        self.device                 = device
        self.train_data             = train_data
        self.val_data               = val_data
        self.prev_checkpoint        = prev_checkpoint
        self.prev_weight            = prev_weight
        self.epoch                  = epoch
        self.save_best_weight       = save_best_weight
        self.checkpoint_freq        = checkpoint_freq
        self.monitor                = monitor
        self.logs                   = dict()
        self.filepath               = filepath
        self.mode                   = min if mode == 'min' else max
        self.metric_best            = +np.inf if mode=='min' else -np.inf
        self.metric_best_epoch      = None


        
    def preprocess(self):
        self.epoch_idx = 1
        self.logs = {'epoch': []}
        self.model = self.model.to(self.device)


    def save_best(self):
        ''' Checks if best metric is found, if so, best weigth is saved'''

        if self.monitor not in self.logs:
            print(f"{self.monitor} not in {self.logs.keys()}")
            return

        if not self.save_best_weight:
            return

        if self.logs[self.monitor][-1] != self.mode(self.logs[self.monitor], self.metric_best):
            return

        fpath = os.path.join(self.filepath, 'best_weight.pth')
        torch.save(self.model.state_dict(), fpath)
        

    def load_model(self):
        ''' Load model '''
        if not self.prev_weight:
            return

        try:
            fpath = os.path.join(self.filepath, 'best_weight.pth')
            self.model.load_state_dict(torch.load(fpath), strict=False)
            print('Previous best model weight loaded')
        except:
            print('Could not load previous best weight')

    
    def save_checkpoint(self):
        fpath = os.path.join(self.filepath, 'model_checkpoint.pkl')
        self.logs['model_weight'] = self.model.state_dict()

        with open(fpath, 'wb') as file:
            pickle.dump(self.logs, file)


    def load_checkpoint(self):
        fpath = os.path.join(self.filepath, 'model_checkpoint.pkl')

        if os.path.exists(fpath):
            with open(fpath, 'rb') as file:
                checkpoint = pickle.load(file)
                self.metric_best = self.mode(checkpoint['logs'][self.monitor])
                self.logs = checkpoint['logs']
                #fresh_start = False

                try:
                    model.load_state_dict(checkpoint['model_weight'], strict=True)
                    print('Checkpoint model loaded')
                except:
                    print('Checkpoint model cannot be loaded')
            
                print("Starting from epoch", checkpoint['epoch'][-1])
                print(f"Best {self.monitor}: {self.metric_best:.4f}")



    def dump_logs(self, logs, to:dict=None, mode:str=''):
        ''' Dumps logs in a dictionary '''

        if mode != '': mode = mode + '_'
        if to is None: to = self.logs

        for k, v in logs.items():
            k = f"{mode}{k}"

            if k not in to:
                to[k] = []
            
            if isinstance(v, torch.Tensor):
                to[k].append(v[-1].item())
            else:
                to[k].append(v[-1])


    def train(self):
        ''' Training caller '''

        self.preprocess()

        # Starting epoch
        while self.epoch_idx <= self.epoch:
            self.logs['epoch'].append(self.epoch_idx)

            # Starting epoch
            self.on_epoch_begin()
            # Training
            self._train()
            # Validation
            if self.val_data:
                self._validate()

            # Progress printing
            self.on_epoch_end()
            self.progbar()
            
            # Saving checkpoint and model
            self.save_checkpoint()
            self.save_best()

            self.epoch_idx += 1


    def _train(self):
        ''' Training function '''

        scaler = torch.cuda.amp.GradScaler(enabled=True)
        batch_log = {}
        self.on_epoch_begin(mode='train')
        
        for x, y in self.train_data:
            self.on_batch_begin()

            x, y = x.to(self.device), y.to(self.device)

            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=True):    
                
                # output is a dictionary
                logs = self.model.forward(x, y)
                batch_loss = logs['loss']

                self.dump_logs(logs, to=batch_log)
                
                scaler.scale(batch_loss).backward()
                scaler.step(self.model.optimizer)
                scaler.update()

            self.on_batch_end()
        
        self.on_epoch_end(mode='train')

        tmp_log = []
        for k in batch_log.keys():
            tmp_log.append(('train_'+k, [sum(batch_log[k]) / len(batch_log[k])]))
        batch_log = dict(tmp_log)
        batch_log['time'] = [self.edtime-self.sttime]
        self.dump_logs(batch_log, mode='')
        


    def _validate(self):
        pass
    

    def on_epoch_begin(self, mode='train'):
        self.sttime = time.time()


    def on_epoch_end(self, mode='train'):
        self.edtime = time.time()
        
        if self.model.scheduler: 
            self.model.scheduler.step()


    def on_batch_begin(self, mode='train'):
        pass


    def on_batch_end(self, mode='train'):
        pass


    def progbar(self, mode='train'):

        print_str = f"Epoch {self.epoch_idx:3d}/{self.epoch}: "

        for k in self.logs.keys():
            if k == 'epoch': continue
            if k == 'model_weight': continue
            print_str += f"{k}: {self.logs[k][-1]:.2f} "
        
        print(print_str)


    

if __name__ == '__main__':

    class Nothing(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scheduler = None
            self.p = torch.nn.Parameter(torch.tensor([0.7]), requires_grad=True)
            self.optimizer = torch.optim.Adam(params=self.parameters())

        def forward(self, x, y=None):
            return {'loss': self.p + x.sum()}

    model = Nothing()

    train_data = [(torch.tensor([5.]), torch.tensor([10.+i])) for i in range(10)]

    trainer = Trainer(model=model, device='cuda', train_data=train_data, epoch=5)

    trainer.train()