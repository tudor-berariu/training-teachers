from collections import OrderedDict
from typing import Optional
from torch import Tensor
import torch.nn as nn
from termcolor import colored as clr
from utils import get_kwargs

"""
import models.classifiers as classifiers  # pylint: disable=unused-import
import models.data_generators as data_generators  # pylint: disable=unused-import
import models.loss_predictors as loss_predictors  # pylint: disable=unused-import
"""


def get_model(module, model_args, *args, **kwargs):
    margs = get_kwargs(model_args)
    print("[MODEL] Initializing a", clr(model_args.name, 'red'), "with", margs)
    return getattr(module, model_args.name)(*args, **margs, **kwargs)


class Student(nn.Module):

    # pylint: disable=arguments-differ
    def forward(self, _data: Tensor,
                _parameters: Optional[OrderedDict]) -> Tensor:
        raise NotImplementedError

    def reset_parameters(self, _state_dict: Optional[OrderedDict]) -> None:
        raise NotImplementedError
