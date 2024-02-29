import importlib
import torch.nn as nn

def get_model(config,device="cpu"):

    script = config["script"] # Name of model script. e.g. SuperPoint.py
    class_name = config["class_name"] # Name of class in model script. e.g. SuperPoint class

    model_script = importlib.import_module(f"superpointprp.models.{script}")
    model = getattr(model_script, class_name)(config)
        
    return model.to(device)

def load_checkpoint(model: nn.Module,
                    pretrained_dict: dict,
                    eval: bool = False):
    """
    Load pretrained checkpoint.
    Input:
        model: nn.Module, model
        pretrained_state_dict: dict, pretrained state dict
    Output:
        model: nn.Module, model
    """
    
    model_state_dict =  model.state_dict()
    pretrained_state = pretrained_dict["model_state_dict"]

    for k,v in pretrained_state.items():
        if k in model_state_dict.keys():
            model_state_dict[k] = v
    
    model.load_state_dict(model_state_dict)
    
    print(f'\033[92m✅ Loaded pretrained model \033[0m')

    return model if not eval else model.eval()