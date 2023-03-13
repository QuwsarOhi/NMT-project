
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import Tensor
from torch.nn import Module
from typing import List, Optional
import torch
import torch.nn.functional as F


class T5(Module):
    '''
    T5 model from: https://huggingface.co/docs/transformers/model_doc/t5
    '''

    def __init__(self, 
                 variant:str="t5-small",
                 max_source_length:int=512, 
                 max_target_length:int=128
                ):

        assert variant in ["t5-small", "t5-base", "t5-large"]

        self.variant = variant
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")


    def forward(self, input:List[str], target:Optional[List[str]]=None) -> Tensor:

        '''
        Will receive input and target string and produce the final output as tensor (not decoded)
        when target is not None, it will give the loss functions with the output as tuple
        '''

        pass


    def predict(self, input:List[str]) -> List[str]:
        
        '''
        Will generate the target output as string
        '''

        pass


if __name__ == '__main__':

    '''
    Implement a tester class similar to T5-old.py
    '''