# hubconf.py
import torch
import torchvision
from timm.models.vision_transformer import vit_base_patch16_224 as _vit_base_patch16_224 # helper function

def PretrainedVit(pretrained=False, **kwargs):
    """Pretrained Vision Transformer model with 16x16 patches and 224x224 input resolution
    pretrained (bool): if True, load pretrained weights
    kwargs: other arguments for the model
    """
    model = _vit_base_patch16_224(**kwargs) # create the model
    if pretrained:
        checkpoint = 'https://github.com/qqmakana/tire_detection/blob/main/Good.ipynb' # URL of the pretrained weights
        state_dict = torch.hub.load_state_dict_from_url(checkpoint, progress=False) # load the state dict
        model.load_state_dict(state_dict) # set the state dict
    return model # return the model

