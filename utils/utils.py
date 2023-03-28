import torch
import numpy as np


def model_summary(model):
    tot_params = 0
    trainable_params = 0
    non_trainable_params = 0

    def prod(l):
        ret = 1
        for ll in l: ret *= ll
        return ret

    print(f"{'Idx':3} | {'Layer':40} | {'Shape':15} | {'Params':10} | {'Trainable'}")
    print("="*70)
    idx = 0

    for name, param in model.named_parameters():
        params = prod(param.size())
        tot_params += params

        if param.requires_grad:
            trainable_params += params
        else:
            non_trainable_params += params

        print(f"{idx:3} | {name[:40]:40} |"
              f" {str(list(param.size()))[:15]:15} |"
              f" {str(params):10} |"
              f" {param.requires_grad}\n")

        idx += 1
        
    print("-"*70)
    print(f"Total Parameters: {tot_params}")
    print(f"Trainable Params: {trainable_params}")
    

if __name__ == '__main__':
    
    import sys, os
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))
    
    from model.T5 import T5
    
    model_summary(T5())