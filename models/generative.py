from typing import Tuple
from functools import reduce
from operator import mul
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_size: Tuple[int, int, int], nz: int):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(reduce(mul, in_size), 400)
        self.linear2 = nn.Linear(400, 400)
        self.mu_linear = nn.Linear(400, nz)
        self.sigma_linear = nn.Linear(400, nz)

    # pylint: disable=arguments-differ
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.mu_linear(x), self.sigma_linear(x)


class Generator(nn.Module):
    def __init__(self, in_size: Tuple[int, int, int], nclasses: int,
                 nz: int, ngf: int) -> None:
        super(Generator, self).__init__()
        nc, _, _ = in_size
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(nz + nclasses, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)
        )

        self.nz = nz
        self.classrange = torch.arange(nclasses).long()
        self.classeye = torch.eye(nclasses)

    # pylint: disable=arguments-differ
    def to(self, device) -> nn.Module:
        self.classrange = self.classrange.to(device)
        self.classeye = self.classeye.to(device)
        return super(Generator, self).to(device)

    # pylint: disable=arguments-differ
    def forward(self, target: torch.Tensor = None,
                nsamples: int = None,
                mean: torch.Tensor = None,
                log_var: torch.Tensor = None):
        device = self.classeye.device
        if target is not None:
            cond = self.classeye[target]
        elif nsamples is not None:
            target = torch.randint(10, (nsamples,), device=device).long()
            cond = self.classeye[target]
        elif mean is not None and log_var is not None:
            target = torch.randint(10, (mean.size(0),), device=device).long()
            cond = self.classeye[target]
        else:
            cond = self.classeye
            target = self.classrange

        batch_size = cond.size(0)
        noise = torch.randn(batch_size, self.nz, device=cond.device)
        if mean is not None and log_var is not None:
            noise = noise * torch.exp(.5 * log_var) + mean
        z = torch.cat((cond, noise), dim=1).unsqueeze(2).unsqueeze(3)
        x = self.generator(z).view(-1, 1, 32, 32)

        return torch.tanh(x), target


class OutputDiscriminator(nn.Module):

    def __init__(self, nclasses: int):
        super(OutputDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(nclasses * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    # pylint: disable=arguments-differ
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
