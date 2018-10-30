from typing import Tuple
from functools import reduce
from operator import mul
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearEncoder(nn.Module):
    def __init__(self, in_size: Tuple[int, int, int],
                 nz: int, nef: int) -> None:
        super(LinearEncoder, self).__init__()
        self.linear1 = nn.Linear(reduce(mul, in_size), nef * 64)
        self.linear2 = nn.Linear(nef * 64, nef * 32)
        self.mu_linear = nn.Linear(nef * 32, nz)
        self.sigma_linear = nn.Linear(nef * 32, nz)

    # pylint: disable=arguments-differ
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.mu_linear(x), self.sigma_linear(x)


class ConvEncoder(nn.Module):
    def __init__(self, in_size: Tuple[int, int, int],
                 nz: int, nef: int) -> None:
        nc, inh, inw = in_size
        if (inh, inw) != (32, 32):
            raise ValueError("Wrong input size. Expected nc x 32 x 32.")

        super(ConvEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef) x 16 x 16
            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*2) x 8 x 8
            nn.Conv2d(nef * 2, nef * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*4) x 4 x 4
        )
        self.mu_linear = nn.Linear(nef * 4 * 16, nz)
        self.sigma_linear = nn.Linear(nef * 4 * 16, nz)

    def forward(self, x):
        x = self.encoder(x).view(x.size(0), -1)
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
                log_var: torch.Tensor = None,
                tmask: torch.Tensor = None):
        device = self.classeye.device
        if target is not None:
            cond = self.classeye[target]
            if tmask is not None:
                cond *= tmask.unsqueeze(1)
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
        z = torch.cat((cond.detach(), noise), dim=1).unsqueeze(2).unsqueeze(3)
        x = self.generator(z)

        return torch.tanh(x), target


class SkipGenerator(nn.Module):
    def __init__(self, in_size: Tuple[int, int, int], nclasses: int,
                 nz: int, ngf: int) -> None:
        super(SkipGenerator, self).__init__()
        nc, _, _ = in_size
        self.pairs = pairs = [
            (nn.Sequential(
                nn.ConvTranspose2d(nz + nclasses, ngf * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)),
             nn.ConvTranspose2d(nz, ngf, 8, 1, 0, bias=False)),
            (nn.Sequential(
                nn.BatchNorm2d(ngf * 3),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 3, ngf, 4, 2, 1, bias=False)),
             nn.ConvTranspose2d(nz, ngf, 16, 1, 0, bias=False)),
            (nn.Sequential(
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False)),
             None)
        ]
        for idx, (layer, skip) in enumerate(pairs):
            setattr(self, f"layer_{idx:d}", layer)
            if skip is not None:
                setattr(self, f"skip_{idx:d}", skip)

        self.nz = nz
        self.classrange = torch.arange(nclasses).long()
        self.classeye = torch.eye(nclasses)

    # pylint: disable=arguments-differ
    def to(self, device) -> nn.Module:
        self.classrange = self.classrange.to(device)
        self.classeye = self.classeye.to(device)
        return super(SkipGenerator, self).to(device)

    # pylint: disable=arguments-differ
    def forward(self, target: torch.Tensor = None,
                nsamples: int = None,
                mean: torch.Tensor = None,
                log_var: torch.Tensor = None,
                tmask: torch.Tensor = None):
        device = self.classeye.device
        if target is not None:
            cond = self.classeye[target]
            if tmask is not None:
                cond *= tmask.unsqueeze(1)
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
        z = torch.cat((cond.detach(), noise), dim=1).unsqueeze(2).unsqueeze(3)
        x = self.model_forward(z, noise.unsqueeze(2).unsqueeze(3))

        return torch.tanh(x), target

    def model_forward(self, z: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        x = z
        for layer, skip in self.pairs:
            x = layer(x)
            if skip is not None:
                x = torch.cat((x, skip(noise)), dim=1)
        return x


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
