from __future__ import absolute_import

from .pcb import *
from .mlfn import *
from .hacnn import *
from .osnet import *
from .osnet_ain import *
from .mudeep import *
from .resnetmid import *

__reid_datasets = [
    'Market-1501',
    'DukeMTMC-reID',
    'MSMT17'
]
__reid_model_factory = {
    # Multi-scale Deep Neural Network:
    # 'mudeep': MuDeep,
    # Residual Network with mid-level features:
    # 'resnet50mid': resnet50mid,
    # Harmonious Attention Convolutional Neural Network:
    'hacnn': HACNN,
    # Part-based Convolutional Baseline:
    # 'pcb_p6': pcb_p6,
    # 'pcb_p4': pcb_p4,
    # Multi-Level Factorisation Net:
    'mlfn': mlfn,
    # Omni-Scale Network:
    'osnet_x1_0': osnet_x1_0,
    'osnet_x0_75': osnet_x0_75,
    'osnet_x0_5': osnet_x0_5,
    'osnet_x0_25': osnet_x0_25,
    'osnet_ibn_x1_0': osnet_ibn_x1_0,  # IBN layer
    # Omni-Scale ain Network:
    'osnet_ain_x1_0': osnet_ain_x1_0,
    # 'osnet_ain_x0_75': osnet_ain_x0_75,
    # 'osnet_ain_x0_5': osnet_ain_x0_5,
    # 'osnet_ain_x0_25': osnet_ain_x0_25
}

def show_available_datasets():
    return __reid_datasets

def show_available_models():
    return list(__reid_model_factory.keys())

def get_model_name(model):
    available_models = list(__reid_model_factory.keys())
    for x in __reid_model_factory.keys():
        if x in model.name:
            return x
    raise KeyError(f"""Unknown model: {model.name}. Must be one of {available_models}""")

def build_model(name, num_classes, loss = 'softmax', pretrained = True, use_gpu = True):
    """A function wrapper for building a model.

    Args:
        name (str): model name.
        num_classes (int): number of training identities.
        loss (str, optional): loss function to optimize the model. Currently
            supports "softmax" and "triplet". Default is "softmax".
        pretrained (bool, optional): whether to load ImageNet-pretrained weights.
            Default is True.
        use_gpu (bool, optional): whether to use gpu. Default is True.

    Returns:
        nn.Module

    Examples::
        >>> from reid import models
        >>> model = models.build_model('resnet50', 751, loss = 'softmax')
    """

    model = __reid_model_factory[name](num_classes = num_classes, loss = loss, pretrained = pretrained, use_gpu = use_gpu)
    
    return model