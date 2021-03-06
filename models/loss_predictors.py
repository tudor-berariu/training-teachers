from collections import OrderedDict
from itertools import chain
from typing import List
from argparse import Namespace
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np

from utils import get_kwargs


class LossLearning(nn.Module):

    def __init__(self,
                 task_model: nn.Module,
                 loss_predictor: Namespace,
                 nprofessors: int) -> None:
        super(LossLearning, self).__init__()
        self.task_model = task_model
        LossPredictor = eval(loss_predictor.name)
        self.professors = []
        self.nprofessors = nprofessors
        for idx in range(nprofessors):
            professor = LossPredictor(task_model, **get_kwargs(loss_predictor))
            self.professors.append(professor)
            setattr(self, f"professor_{idx:d}", professor)

    def forward(self, params: OrderedDict) -> List[Tensor]:
        return [prof(params) for prof in self.professors]

    def professor_parameters(self):
        return chain(*[prof.parameters() for prof in self.professors])

    def eval_student(self, student):
        params = OrderedDict([(name, param.unsqueeze(0))
                              for (name, param) in student.named_parameters()])
        return sum(professor(params) for professor in self.professors)


class CorrLossPredictor(nn.Module):

    def __init__(self, task_model: nn.Module,
                 inter_layer: bool,
                 width: int) -> None:
        super(CorrLossPredictor, self).__init__()
        total_features = 0
        self.inter_layer = inter_layer
        self.width = width
        self.per_layer = OrderedDict({})
        self.pairs = []
        last_name, last_nout = None, None
        for idx, (name, param) in enumerate(task_model.named_parameters()):
            nin = param.nelement()
            nout = np.int(np.log(nin)) * width
            self.per_layer[name] = layer = nn.Linear(nin, nout, bias=False)
            setattr(self, f"layer_{idx:d}", layer)
            total_features += nout + nout * nout

            if "bias" not in name and inter_layer:
                if last_name is not None:
                    total_features += last_nout * nout
                    self.pairs.append((last_name, name))
                last_nout = nout
                last_name = name
        self.fc = nn.Sequential(nn.Linear(total_features, 256),
                                nn.ReLU(),
                                nn.Linear(256, 16))

    def forward(self, named_params: OrderedDict) -> Tensor:
        # This model assumes parameters have a batch dimension
        features = []
        layer_and_params = zip(self.per_layer.items(), named_params.items())
        last_name, last_proj = None, None
        for named_layer, named_param in layer_and_params:
            param_name, param = named_param
            layer_name, layer = named_layer
            assert param_name == layer_name

            batch_size = param.size(0)
            param = param.view(batch_size, -1)
            crt_proj = layer(param)
            features.append(crt_proj)
            features.append(torch.bmm(crt_proj.unsqueeze(2),  # outer product
                                      crt_proj.unsqueeze(1)))

            if "bias" not in param_name and self.inter_layer:
                if (last_name, param_name) in self.pairs:
                    features.append(torch.bmm(last_proj.unsqueeze(2),
                                              crt_proj.unsqueeze(1)))
                last_proj, last_name = crt_proj, param_name
        all_features = tuple(f.view(batch_size, -1) for f in features)
        errors = self.fc(torch.cat(all_features, dim=1)).squeeze(1)
        return errors.pow(2).sum(dim=1)


class MLPLossPredictor(nn.Module):

    def __init__(self, task_model: nn.Module,
                 inter_layer: bool,
                 width: int) -> None:
        super(MLPLossPredictor, self).__init__()
        total_features = 0
        self.per_layer = OrderedDict({})
        for idx, (name, param) in enumerate(task_model.named_parameters()):
            nfeatures = param.size(0)
            nin = param.nelement() // nfeatures
            nout = width * np.int(np.sqrt(nin))
            total_features += nout * nfeatures
            self.per_layer[name] = layer = nn.Linear(nin, nout, bias=False)
            setattr(self, f"layer_{idx:d}", layer)

        self.inter_layer = inter_layer
        if inter_layer:
            self.per_pairs = OrderedDict({})
            last_param, last_name, last_idx = None, None, None
            for idx, (name, param) in enumerate(task_model.named_parameters()):
                if "bias" not in name:
                    if last_param is None:
                        last_param = param.view(param.size(0), -1)
                        last_name = name
                        last_idx = idx
                    else:
                        crt_param = param.view(param.size(0), -1)
                        n_1, d_1 = last_param.size()
                        n_2, d_2 = crt_param.size()
                        nout = width * np.int(np.sqrt(n_1 * n_2))
                        layer = nn.Linear(n_1 * n_2, nout, bias=False)
                        total_features += nout
                        corr = nn.Parameter(torch.randn(1, d_2, d_1).to(param.device) /
                                            (d_1 + d_2))
                        self.per_pairs[(last_name, name)] = (corr, layer)
                        setattr(self, f"corr_{last_idx:d}_{idx:d}", corr)
                        setattr(self, f"layers_{last_idx:d}_{idx:d}", layer)

                        last_param, last_name, last_idx = crt_param, name, idx

        self.fc = nn.Sequential(nn.Linear(total_features, 256),
                                nn.ReLU(),
                                nn.Linear(256, 1),
                                nn.LeakyReLU())

    def forward(self, named_params: OrderedDict) -> Tensor:
        features = []
        layer_and_params = zip(self.per_layer.items(), named_params.items())
        last_name, last_param = None, None
        for named_layer, named_param in layer_and_params:
            param_name, param = named_param
            layer_name, layer = named_layer
            assert param_name == layer_name

            batch_size, nout, *_other = param.size()
            param = param.view(batch_size * nout, -1)
            features.append(layer(param).view(batch_size, -1))

            if self.inter_layer and "bias" not in param_name:
                crt_param = param.view(batch_size, nout, -1)
                if (last_name, param_name) in self.per_pairs:
                    d_1, d_2 = last_param.size(2), crt_param.size(2)
                    corr, layer = self.per_pairs[(last_name, param_name)]
                    left = torch.bmm(last_param, corr.expand(batch_size, d_2, d_1).transpose(1, 2))
                    features.append(layer(torch.bmm(left, crt_param.transpose(1, 2)).view(batch_size, -1)))
                last_param, last_name = crt_param, param_name

        return self.fc(torch.cat(tuple(features), dim=1)).squeeze(1)
