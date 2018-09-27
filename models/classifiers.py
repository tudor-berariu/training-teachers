from typing import List
from collections import OrderedDict
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F



class MLP(nn.Module):
    def __init__(self,
                 nin: int, nout: int,
                 units: List[int],
                 use_bias: bool = False,
                 use_dropout: bool = False) -> None:
        super(MLP, self).__init__()
        units = units + [nout]
        params = OrderedDict({})

        for idx, nout in enumerate(units):
            name = f"l{idx:d}_weight"
            weight = torch.empty(nout, nin)
            nn.init.kaiming_uniform_(weight, a=0, mode='fan_out',
                                     nonlinearity='relu')
            weight = nn.Parameter(weight, requires_grad=True)
            setattr(self, name, weight)
            params[name] = weight

            if use_bias:
                name = f"l{idx:d}_bias"
                bias = nn.Parameter(torch.zeros(nout), requires_grad=True)
                setattr(self, name, bias)
                params[name] = bias

            nin = nout

        self.nlayers = len(units)
        self.params = params
        self.use_dropout = use_dropout

        print(f"Initialized MLP with {self.nlayers:d} layers.")

    def forward(self, x: Tensor, params: OrderedDict = None) -> Tensor:
        if params is None:
            params = self.params
        x = x.view(x.size(0), -1)
        x = F.linear(x, params["l0_weight"], params.get("l0_bias", None))
        for idx in range(1, self.nlayers):
            x = F.relu(x)
            if self.use_dropout:
                x = F.dropout(x, training=self.training)
            wname, bname = f"l{idx:d}_weight", f"l{idx:d}_bias"
            x = F.linear(x, params[wname], params.get(bname, None))
        return x

"""
class ConvNet(nn.Module):

    def __init__(self, use_bias: bool = False) -> None:
        super(ConvNet, self).__init__()

        nin = 1
        features = [10, 20]
        kernels = [5, 5]

        my_params = OrderedDict({})

        for idx, (nout, k) in enumerate(zip(features, kernels)):
            name = f"c{idx:d}_weight"
            weight = torch.empty(nout, nin, k, k)
            nn.init.kaiming_uniform_(weight, a=0, mode='fan_out',
                                     nonlinearity='relu')
            weight = nn.Parameter(weight, requires_grad=True)
            my_params[name] = weight
            setattr(self, name, weight)
            if use_bias:
                name = f"c{idx:d}_bias"
                bias = nn.Parameter(torch.zeros(nout), requires_grad=True)
                my_params[name] = bias
                setattr(self, name, bias)

            nin = nout

        nin = 320
        units = [50, 10]

        for idx, nout in enumerate(units):
            name = f"l{idx:d}_weight"
            weight = torch.empty(nout, nin)
            nn.init.kaiming_uniform_(weight, a=0, mode='fan_out',
                                     nonlinearity='relu')
            weight = nn.Parameter(weight, requires_grad=True)
            setattr(self, name, weight)
            my_params[name] = weight

            if use_bias:
                name = f"l{idx:d}_bias"
                bias = nn.Parameter(torch.zeros(nout), requires_grad=True)
                setattr(self, name, bias)
                my_params[name] = bias

            nin = nout

        self.my_params = my_params

    def forward(self, x: Tensor, params: OrderedDict=None) -> Tensor:
        if params is None:
            params = self.my_params
        x = F.relu(F.max_pool2d(F.conv2d(x, params["c0_weight"], params.get("c0_bias", None)), 2))
        x = F.relu(F.max_pool2d(F.conv2d(x, params["c1_weight"], params.get("c1_bias", None)), 2))
        x = x.view(-1, 320)
        x = F.relu(F.linear(x, params["l0_weight"], params.get("l0_bias", None)))
        x = F.dropout(x, training=self.training)
        x = F.linear(x, params["l1_weight"], params.get("l1_bias", None))
        return F.log_softmax(x, dim=1)
"""
