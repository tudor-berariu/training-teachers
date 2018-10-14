from typing import Iterator, Tuple

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F


def collapse(parameters: Iterator[Tensor], detach: bool = False):
    # TODO: collapse batches of parameters
    if detach:
        return torch.cat(tuple(t.clone().view(-1).detach() for t in parameters))

    return torch.cat(tuple(t.view(-1) for t in parameters))


def l2(parameters: Iterator[Tensor]) -> Tensor:
    return sum(param.pow(2).sum() for param in parameters)


def cos(parameters: Iterator[Tensor], targets: Iterator[Tensor]):
    flat_parameters = collapse(parameters).unsqueeze(0)
    flat_targets = collapse(targets, detach=True).unsqueeze(0)
    return F.cosine_embedding_loss(flat_parameters, flat_targets,
                                   torch.Tensor([1]).to(flat_parameters.device))


def mse(params: Iterator[Tensor], targets: Iterator[Tensor]) -> Tensor:
    mse_loss = nn.MSELoss(reduction='sum')
    full_sum = 0
    nelements = 0
    for y, t in zip(params, targets):
        nelements += y.nelement()
        full_sum += mse_loss(y, t.detach())
    return full_sum / nelements
