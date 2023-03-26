
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import Tensor
from torch.nn import Module
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F


class T5(Module):
    '''
    T5 model from: https://huggingface.co/docs/transformers/model_doc/t5
    '''

    def __init__(self, 
                 variant:str="t5-small",
                 max_source_length:int=512, 
                 max_target_length:int=128,
                 optimizer_config:dict={},
                ):

        # Assertions
        assert variant in ["t5-small", "t5-base", "t5-large"]

        super().__init__()

        self.variant            = variant
        self.max_source_length  = max_source_length
        self.max_target_length  = max_target_length

        # Tokenizer & model
        self.tokenizer          = T5Tokenizer.from_pretrained(variant)
        self.model              = T5ForConditionalGeneration.from_pretrained(variant)
        
        # Optimizer
        self.optimizer          = torch.optim.AdamW(self.parameters(), **optimizer_config)
        
        # Scheduler
        self.scheduler          = None


    def tokenize(self, input:List[str]):

        out = self.tokenizer(input, max_length=self.max_source_length,
                             truncation=False, padding=True, 
                             return_tensors="pt")

        return out.input_ids, out.attention_mask


    def forward(self, input:List[str], label:Optional[List[str]]=None) -> Tuple[Tensor, Optional[Tensor]]:

        '''
        Will receive input and target string and produce the final output as tensor (not decoded)
        when target is not None, it will give the loss functions with the output as tuple
        '''
        
        input_ids, input_masks = self.tokenize(input)
        
        if label is not None:
            label_ids, label_masks = self.tokenize(label)    
            output = self.model(input_ids=input_ids, labels=label_ids)
            return output.logits, output.loss
            
        return self.model.generate(input_ids=input_ids,
                                   max_length=128,
                                   max_new_tokens=20), None



    def predict(self, input:List[str]) -> List[str]:
        
        '''
        Will generate the target output as string
        '''
        
        logits, loss = self.forward(input=input)
        
        return self.tokenizer.batch_decode(logits, skip_special_tokens=True)
        


if __name__ == '__main__':

    '''
    Implement a tester class similar to T5-old.py to test if it works
    '''
    
    model = T5('t5-small')

    inputs = [
        "translate English to German: Thank you so much, Chris.",
        "translate English to German: I have been blown away by this conference, and I want to thank all of you for the many nice comments about what I had to say the other night.",
        "translate German to English: Es ist mir wirklich eine Ehre, zweimal auf dieser Bühne stehen zu dürfen. Tausend Dank dafür.",
    ]

    targets = [
        "Vielen Dank, Chris.",
        "Ich bin wirklich begeistert von dieser Konferenz, und ich danke Ihnen allen für die vielen netten Kommentare zu meiner Rede vorgestern Abend.",
        "And it's truly a great honor to have the opportunity to come to this stage twice; I'm extremely grateful.",
    ]

    logits, loss = model.forward(inputs, targets)
    print('Model forward')
    print('logits: ', logits)
    print('loss: ', loss)
    
    outputs = model.predict(inputs)
    
    print('OUTPUT')
    print(outputs)
    #for (inp, out), tar in zip(zip(inputs, outputs), targets):
    #    print(f"Input: {inp}\nOutput: {out}\nTarget: {tar}\n")


