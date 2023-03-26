from torch.utils.data import DataLoader, Dataset
import datasets, os
from typing import Tuple


class DataGen(Dataset):

    def __init__(self, 
                 config_id:int,                 # which config id (language mapping) to be used
                 verbose:bool=True,             # verbose logging
                 data_split:str='train',         # data split [train, test, val]
                 cache_dir:str='../dataset',    # path to where the data will be saved
                ):
    

        # Dataset language maps
        self.config_name = [
            'iwslt2017-en-it', 'iwslt2017-en-nl', 'iwslt2017-en-ro', 
            'iwslt2017-it-en', 'iwslt2017-it-nl', 'iwslt2017-it-ro', 
            'iwslt2017-nl-en', 'iwslt2017-nl-it', 'iwslt2017-nl-ro', 
            'iwslt2017-ro-en', 'iwslt2017-ro-it', 'iwslt2017-ro-nl', 
            'iwslt2017-ar-en', 'iwslt2017-de-en', 'iwslt2017-en-ar', 
            'iwslt2017-en-de', 'iwslt2017-en-fr', 'iwslt2017-en-ja', 
            'iwslt2017-en-ko', 'iwslt2017-en-zh', 'iwslt2017-fr-en', 
            'iwslt2017-ja-en', 'iwslt2017-ko-en', 'iwslt2017-zh-en'
        ]

        assert data_split in ['train', 'test', 'validation']

        # Vervosity
        self.verbose = verbose
        # Cache directory of dataset
        self.cache_dir = cache_dir
        # Language mapping specification
        self.config = self.config_name[config_id]
        # Language splits: ['train', 'test', 'validation']
        self.data_split = data_split

        # create filepath
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.dataset = datasets.load_dataset("iwslt2017", 
                                             self.config,
                                             cache_dir=self.cache_dir)[self.data_split]

        # Input and output language short-code
        [self.in_lang, self.out_lang] = self.config[10:].split('-')

        if self.verbose:
            print(f"Loaded config : {self.config} -> {self.data_split} split")
    

    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, idx):
        # dataset is in dict format
        # {'translation': {'de': 'Vielen Dank, Chris.', 'en': 'Thank you so much, Chris.'}}
        data = self.dataset[idx]['translation']

        inp = data[self.in_lang]
        out = data[self.out_lang]
        
        if self.verbose:
            print(f"{self.in_lang} : {inp}\n{self.out_lang} : {out}")

        return inp, out



def get_dataset(batch_size, drop_last=True, shuffle=True, num_workers=4, 
                pin_memory=True, verbose=False) -> Tuple[DataLoader, DataLoader, DataLoader]:


    train_data = DataLoader(DataGen(config_id=15, verbose=verbose, data_split='train'),
                            batch_size=batch_size, shuffle=shuffle, 
                            drop_last=drop_last, num_workers=num_workers, 
                            pin_memory=pin_memory)
    
    val_data = DataLoader(DataGen(config_id=15, verbose=verbose, data_split='validation'),
                          batch_size=batch_size, shuffle=shuffle, 
                          drop_last=drop_last, num_workers=num_workers, 
                          pin_memory=pin_memory)
    
    test_data = DataLoader(DataGen(config_id=15, verbose=verbose, data_split='train'),
                           batch_size=batch_size, shuffle=shuffle, 
                           drop_last=drop_last, num_workers=num_workers, 
                           pin_memory=pin_memory)
    
    return train_data, val_data, test_data



if __name__ == '__main__':
    data = DataGen(config_id=15, verbose=True)
    print(data[0])
