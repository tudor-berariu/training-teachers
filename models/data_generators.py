from typing import Tuple
import torch
import torch.nn as nn


class SyntheticDataGenerator(nn.Module):
    def __init__(self, in_size: Tuple[int, int, int], nclasses: int,
                 nz: int, ngf: int) -> None:
        super(SyntheticDataGenerator, self).__init__()
        nc, _, _ = in_size
        self.data_net = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz + nclasses, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )
        self.nz = nz
        self.classrange = torch.arange(nclasses).long()
        self.classeye = torch.eye(nclasses)

    def to(self, device) -> nn.Module:
        self.classrange = self.classrange.to(device)
        self.classeye = self.classeye.to(device)
        return super(SyntheticDataGenerator, self).to(device)

    def forward(self, targets: torch.Tensor = None, nsamples: int = None):
        device = self.classeye.device
        if targets is None:
            targets = self.classrange
            cond = self.classeye
        elif nsamples is not None:
            targets = torch.randint(10, (nsamples,), device=device).long()
            cond = self.classeye[targets]
        else:
            cond = self.classeye[targets]
        batch_size = cond.size(0)
        noise = torch.randn(batch_size, self.nz, device=cond.device)
        z = torch.cat((cond, noise), dim=1).unsqueeze(2).unsqueeze(3)
        return self.data_net(z), targets


class SyntheticDataDiscriminator(nn.Module):

    def __init__(self, nclasses: int):
        super(SyntheticDataDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(nclasses * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
