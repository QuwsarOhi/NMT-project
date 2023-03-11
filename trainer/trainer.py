from torch.nn import Module
from torch.utils.data import DataLoader
import torch, time

class Trainer:

    def __init__(self, 
                 model:Module,                          # the model to train
                 device:str,                            # device to train on
                 epoch:int,                             # training epoch
                 train_data:DataLoader,                 # training dataset
                 val_data:DataLoader=None,              # validation dataset
                 prev_checkpoint:str='',              # start from the previous checkpoint (path)
                 prev_weight:str='',                  # load previous weight (path)
                ):
        
        self.model = model
        self.device = device
        self.train_data = train_data
        self.val_data = val_data
        self.prev_checkpoint = prev_checkpoint
        self.prev_weight = prev_weight
        self.epoch = epoch

        if prev_checkpoint != '' or prev_weight != '':
            raise NotImplementedError

        
    def preprocess(self):
        self.epoch_idx = 1
        self.logs = {'epoch': []}
        self.model = self.model.to(self.device)


    def dump_logs(self, logs, to:dict=None, mode:str=''):
        ''' Dumps logs in a dictionary '''

        if mode != '': mode = mode + '_'
        if to is None: to = self.logs

        for k, v in logs.items():
            if k not in self.logs:
                self.logs[f"{mode}{k}"] = []
            
            if isinstance(v, torch.Tensor):
                self.logs[k].append(v.item())
            else:
                self.logs[k].append(v)


    def train(self):
        ''' Training caller '''

        self.preprocess()

        while self.epoch_idx <= self.epoch:
            self.logs['epoch'].append(self.epoch_idx)
            self.on_epoch_begin()
            self._train()

            if self.val_data:
                self._validate()

            self.on_epoch_end()
            self.progbar()

            self.epoch_idx += 1


    def _train(self):
        ''' Training function '''

        scaler = torch.cuda.amp.GradScaler(enabled=True)
        batch_log = {}
        
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
        

        for k in batch_log.keys():
            batch_log[k] = sum(batch_log[k]) / len(batch_log[k])
        self.dump_logs(batch_log, mode='train')
        


    def _validate(self):
        pass
    

    def on_epoch_begin(self):
        self.train_sttime = time.time()


    def on_epoch_end(self):
        self.train_edtime = time.time()
        
        if self.model.scheduler: 
            self.model.scheduler.step()


    def on_batch_begin(self):
        pass


    def on_batch_end(self):
        pass

    def progbar(self):

        print_str = f"Epoch {self.epoch_idx:3d}/{self.epoch}: "

        for k in self.logs.keys():
            if k == 'epoch': continue
            print_str += f"{k}: {self.logs[k][-1]:.2f} "
        
        print(print_str)


    

if __name__ == '__main__':

    class Nothing(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scheduler = None
            self.p = torch.nn.Parameter(torch.Tensor(1))
            self.optimizer = torch.optim.Adam(params=self.parameters())

        def forward(self, x, y=None):
            return {'loss': self.p*x.sum()}

    model = Nothing()

    train_data = [(torch.Tensor(10), torch.Tensor(10)) for i in range(10)]

    trainer = Trainer(model=model, device='cuda', train_data=train_data, epoch=5)

    trainer.train()