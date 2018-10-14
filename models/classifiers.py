from functools import reduce
from operator import mul
from typing import List, Tuple
from collections import OrderedDict
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


# TODO: add batch normalization


class MLP(nn.Module):
    def __init__(self,
                 nin: Tuple[int], nout: int,
                 units: List[int],
                 use_bias: bool = False,
                 use_dropout: bool = False) -> None:
        super(MLP, self).__init__()
        units = units + [nout]
        params = OrderedDict({})

        nin = reduce(mul, nin)

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

    def reset_weights(self) -> None:
        for idx in range(self.nlayers):
            wname, bname = f"l{idx:d}_weight", f"l{idx:d}_bias"
            weight, bias = self.params[wname], self.params.get(bname, None)
            nn.init.kaiming_uniform_(weight.data, a=0, mode='fan_out',
                                     nonlinearity='relu')
            if bias is not None:
                bias.data.fill_(0)

    def forward(self, x: Tensor, params: OrderedDict=None) -> Tensor:
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


class ConvNet(nn.Module):

    def __init__(self,
                 nin: Tuple[int, int, int], nout: int,
                 channels: List[int],
                 units: List[int],
                 kernels: List[int] = None,
                 strides: List[int] = None,
                 use_bias: bool = False,
                 use_dropout: bool = True) -> None:
        super(ConvNet, self).__init__()

        nclasses = nout
        nin, h, w = nin
        kernels = [5 for c in channels] if kernels is None else kernels
        strides = [1 for c in channels] if strides is None else strides

        params = OrderedDict({})

        for idx, (nout, k, s) in enumerate(zip(channels, kernels, strides)):
            name = f"c{idx:d}_weight"
            weight = torch.empty(nout, nin, k, k)
            nn.init.kaiming_uniform_(weight, a=0, mode='fan_out',
                                     nonlinearity='relu')
            weight = nn.Parameter(weight, requires_grad=True)
            params[name] = weight
            setattr(self, name, weight)
            if use_bias:
                name = f"c{idx:d}_bias"
                bias = nn.Parameter(torch.zeros(nout), requires_grad=True)
                params[name] = bias
                setattr(self, name, bias)

            nin, h, w = nout, (h - k) // s + 1, (w - k) // s + 1
            h, w = h // 2, w // 2  # from pooling

        nin = nin * h * w
        units = units + [nclasses]

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

        self.nconv = len(channels)
        self.nlinear = len(units)
        self.params = params
        self.use_dropout = use_dropout

        print(f"[MODEL] Initialized Conv with {self.nconv:d} conv layers + "
              f"{self.nlinear:d} linear layers.")

    def reset_weights(self) -> None:
        for idx in range(self.nconv):
            wname, bname = f"c{idx:d}_weight", f"c{idx:d}_bias"
            weight, bias = self.params[wname], self.params.get(bname, None)
            nn.init.kaiming_uniform_(weight.data, a=0, mode='fan_out',
                                     nonlinearity='relu')
            if bias is not None:
                bias.data.fill_(0)
        for idx in range(self.nlinear):
            wname, bname = f"l{idx:d}_weight", f"l{idx:d}_bias"
            weight, bias = self.params[wname], self.params.get(bname, None)
            nn.init.kaiming_uniform_(weight.data, a=0, mode='fan_out',
                                     nonlinearity='relu')
            if bias is not None:
                bias.data.fill_(0)

    def forward(self, x: Tensor, params: OrderedDict=None) -> Tensor:
        if params is None:
            params = self.params

        for idx in range(self.nconv):
            wname, bname = f"c{idx:d}_weight", f"c{idx:d}_bias"
            weight, bias = self.params[wname], self.params.get(bname, None)
            x = F.conv2d(x, weight, bias)
            x = F.relu(F.max_pool2d(x, 2))

        x = x.view(x.size(0), -1)

        x = F.linear(x, params["l0_weight"], params.get("l0_bias", None))
        for idx in range(1, self.nlinear):
            x = F.relu(x)
            if self.use_dropout:
                x = F.dropout(x, training=self.training)
            wname, bname = f"l{idx:d}_weight", f"l{idx:d}_bias"
            x = F.linear(x, params[wname], params.get(bname, None))
        return x
