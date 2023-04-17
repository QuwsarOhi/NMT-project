import torch, os
import json
from T5 import T5


if __name__ == '__main__':

    '''
    Implement a tester class similar to T5-old.py to test if it works
    '''
    
    ############################################################
    with open("config.json", "r") as f:
        config = json.load(f)
    
    model = T5('t5-small').cuda()
    print("Model weight", config["weight"])
    model.load_state_dict(torch.load(config["weight"]))
    ############################################################

    inputs = ["translate Italian to French: There are 7 continents on planet earth. Australia is the largest continent and a country by itself."]
    
    with torch.inference_mode():
        outputs = model.predict(inputs)
    
    print("\nINPUT\n",inputs[0])
    print("\nOUTPUT\n",outputs[0])



