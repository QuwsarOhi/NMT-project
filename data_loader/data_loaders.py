from torchvision import datasets, transforms
from base import BaseDataLoader

class IWSLT_DataGen(torch.utils.data.Dataset):

    def __init__(self, cache_dir: str, config_id: int=15, verbose: bool=True,
                 data_split: str='train'):
        
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



if __name__ == '__main__':

    dataset = IWSLT_DataGen(cache_dir='../datasets/')