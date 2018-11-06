from typing import Tuple
from functools import reduce
from operator import mul
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


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


class GenericGenerator(nn.Module):
    def __init__(self,
                 in_size: Tuple[int, int, int],
                 nclasses: int,
                 nz: int,
                 ngf: int,
                 nperf: int = 0) -> None:
        super(GenericGenerator, self).__init__()
        self.in_size = in_size
        self.nclasses = nclasses
        self.nz = nz
        self.ngf = ngf
        self.nperf = nperf

        self.classrange = torch.arange(nclasses, dtype=torch.long)
        self.classeye = torch.eye(nclasses)

        if nperf > 0:
            self.perf_features = dist.Normal(torch.linspace(0, 100, nperf), 3)
        else:
            self.perf_features = None

    def to(self, device):
        self.classrange = self.classrange.to(device)
        self.classeye = self.classeye.to(device)
        return super(GenericGenerator, self).to(device)

    def forward(self,
                target: torch.Tensor = None,
                nsamples: int = None,
                mean: torch.Tensor = None,
                log_var: torch.Tensor = None,
                tmask: torch.Tensor = None,
                perf: float = None):
        device = self.classeye.device
        if target is None:
            if mean is not None and log_var is not None:
                nsamples = mean.size(0)
            elif nsamples is None:
                raise RuntimeError("Specify one of: nsamples, (mean, log_var)")

            target = torch.randint(self.nclasses, (nsamples,),
                                   device=device, dtype=torch.long)

        cond = self.classeye[target]
        if tmask is not None:
            cond *= tmask.unsqueeze(1)

        batch_size = cond.size(0)
        noise = torch.randn(batch_size, self.nz, device=device)
        if mean is not None and log_var is not None:
            noise = noise * torch.exp(.5 * log_var) + mean

        if self.perf_features is not None:
            pf = self.perf_features.log_prob(perf).exp()\
                     .unsqueeze(0).repeat(batch_size, 1).to(device)
            all_latent = (cond, noise, pf)
        else:
            all_latent = (cond, noise)

        z = torch.cat(all_latent, dim=1).unsqueeze(2).unsqueeze(3)
        x = self._model_forward(z, noise.unsqueeze(2).unsqueeze(3))

        return torch.tanh(x), target

    def _model_forward(self, z, noise):
        raise NotImplementedError


class ConvGenerator(GenericGenerator):
    def __init__(self,
                 in_size: Tuple[int, int, int],
                 nclasses: int,
                 nz: int,
                 ngf: int,
                 nperf: int = 0) -> None:
        super(ConvGenerator, self).__init__(in_size, nclasses, nz, ngf, nperf)
        nc, _, _ = in_size
        ninput = nz + nclasses + nperf

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(ninput, ngf * 4, 4, 1, 0, bias=False),
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

    def _model_forward(self, z, _noise):
        return self.generator(z)


class SkipGenerator(GenericGenerator):
    def __init__(self,
                 in_size: Tuple[int, int, int],
                 nclasses: int,
                 nz: int,
                 ngf: int,
                 nperf: int = 0) -> None:
        super(SkipGenerator, self).__init__(in_size, nclasses, nz, ngf, nperf)
        nc, _, _ = in_size
        ninput = nz + nclasses + nperf
        self.pairs = pairs = [
            (nn.Sequential(
                nn.ConvTranspose2d(ninput, ngf * 4, 4, 1, 0, bias=False),
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

    def _model_forward(self, z, noise):
        x = z
        for layer, skip in self.pairs:
            x = layer(x)
            if skip is not None:
                x = torch.cat((x, skip(noise)), dim=1)
        return x


class OutputDiscriminator(nn.Module):

    def __init__(self, nclasses: int, use_labels: bool=False):
        super(OutputDiscriminator, self).__init__()
        in_no = (nclasses * 2) if use_labels else nclasses
        self.model = nn.Sequential(
            nn.Linear(in_no, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
