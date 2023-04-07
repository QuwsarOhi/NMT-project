from torch.utils.data import DataLoader, Dataset
import datasets, os
from typing import Tuple, List
from torch.utils.data import ConcatDataset

datasets.disable_progress_bar()


class DataGen(Dataset):

    def __init__(self, 
                 config_id:int,                 # which config id (language mapping) to be used
                 verbose:bool=True,             # verbose logging
                 data_split:str='train',         # data split [train, test, val]
                 cache_dir:str='../dataset',    # path to where the data will be saved
                ):
    

        # Dataset language maps
        self.config_name = [
            'iwslt2017-en-it', 'iwslt2017-it-en',
            'iwslt2017-de-en', 'iwslt2017-en-de',
            'iwslt2017-en-nl', 'iwslt2017-nl-en',
            'iwslt2017-en-ro', 'iwslt2017-ro-en',
            'iwslt2017-fr-en', 'iwslt2017-en-fr',
            # Non-english mappings
            'iwslt2017-it-nl', 'iwslt2017-nl-it',
            'iwslt2017-ro-nl', 'iwslt2017-nl-ro', 
            'iwslt2017-ro-it', 'iwslt2017-it-ro', 
            #'iwslt2017-ko-en', 'iwslt2017-en-ko',
            #'iwslt2017-en-ja', 'iwslt2017-ja-en',
            #'iwslt2017-zh-en', 'iwslt2017-en-zh',
            #'iwslt2017-ar-en', 'iwslt2017-en-ar', 
        ]

        assert data_split in ['train', 'test', 'validation']

        # language maps
        self.lang_map = {
            #"ko": "Korean",
            "en": "English",    # 
            "nl": "Dutch",      #
            #"ja": "Japanese",
            #"ar": "Arabic",
            "fr": "French",     #
            "it": "Italian",    #
            "ro": "Romanian",   #
            #"zh": "Chinese",
            "de": "German"      #
        }
        
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
                                             keep_in_memory=True,
                                             cache_dir=self.cache_dir)[self.data_split]

        # Input and output language short-code
        [self.in_lang, self.out_lang] = self.config[10:].split('-')
        self.frm = self.lang_map[self.in_lang]
        self.to = self.lang_map[self.out_lang]

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
        
        inp = f"translate {self.frm} to {self.to}: {inp}"
        
        if self.verbose:
            print(f"{self.in_lang} : {inp}\n{self.out_lang} : {out}")

        return inp, out


class DataCollection(Dataset):
    '''
    Although the ConcatDataset can merge datasets into one
    it does binary search each time its called making it 
    unefficient.
    So here is an memorized version of ConcatDataset.
    It just saves the indices and fetches the indices 
    on demand.
    '''
    
    def __init__(self, data=List[Dataset]):
        
        super().__init__()
        
        self.indices        = []
        self.datasets       = data
        self.n_datasets     = len(self.datasets)
        
        for i in range(self.n_datasets):
            for j in range(len(self.datasets[i])):
                self.indices.append((i, j))
        
        
    def __len__(self) -> int:
        return len(self.indices)
    
    
    def __getitem__(self, index) -> Tuple[List[str], List[str]]:
        i, j = self.indices[index]
        return self.datasets[i][j]
        
        

def get_dataset(batch_size, ids=10, drop_last=True, num_workers=4, 
                pin_memory=True, cache_dir='') -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_data = []
    val_data = []
    test_data = []
    
    if cache_dir:
        cache_dir = '../dataset'

    for i in range(ids):
        train_data.append(DataGen(config_id=i, verbose=False, data_split='train', cache_dir=cache_dir))
        val_data.append(DataGen(config_id=i, verbose=False, data_split='validation', cache_dir=cache_dir))
        test_data.append(DataGen(config_id=i, verbose=False, data_split='test', cache_dir=cache_dir))

    train_data = DataCollection(train_data)
    val_data = DataCollection(val_data)
    test_data = DataCollection(test_data)
    
    train_data = DataLoader(train_data,
                            batch_size=batch_size, shuffle=True, 
                            drop_last=drop_last, num_workers=num_workers, 
                            pin_memory=pin_memory)
    
    val_data = DataLoader(val_data,
                          batch_size=batch_size, shuffle=False, 
                          drop_last=drop_last, num_workers=num_workers, 
                          pin_memory=pin_memory)
    
    test_data = DataLoader(test_data,
                           batch_size=batch_size, shuffle=False, 
                           drop_last=drop_last, num_workers=num_workers, 
                           pin_memory=pin_memory)
    
    return train_data, val_data, test_data



if __name__ == '__main__':
    data = DataGen(config_id=0, #15 
                   verbose=False)
    print(data[0])
