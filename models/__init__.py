from collections import OrderedDict
from torch import Tensor
import torch.nn as nn


class Student(nn.Module):

    def forward(self, _data: Tensor,
                _parameters: OrderedDict,
                _batches: bool = False) -> Tensor:
        raise NotImplementedError

    def reset_parameters(self, _state_dict: OrderedDict) -> None:
        raise NotImplementedError


class Professor(nn.Module):

    def discriminate(self, _real_output, _fake_output, target) -> Tensor:
        raise NotImplementedError
