from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Student, Professor
from utils import get_kwargs


class GenerativeProfessor(Professor):
    def __init__(self, llargs: Namespace) -> None:
        super(GenerativeProfessor, self).__init__()
        data_generator_cfg = llargs.data_generator
        DataGenerator = eval(data_generator_cfg.name)
        self.professor = DataGenerator(**get_kwargs(data_generator_cfg))
        self.crt_device = None
        self.eval_samples = llargs.eval_samples

    def forward(self, targets):
        return self.professor(targets)

    def eval_student(self, student: Student) -> torch.Tensor:
        if self.eval_samples is not None:
            target = torch.randint(10, (self.eval_samples,)).long()
            if self.crt_device:
                target = target.to(self.crt_device)
        else:
            target = None
        data, target = self.professor(targets=target)
        output = student(data)
        return F.cross_entropy(output, target)

    def to(self, device):
        self.crt_device = device
        self.professor.to(device)
        return super(GenerativeProfessor, self).to(device)


class SyntheticDataGenerator(nn.Module):
    def __init__(self, nz: int, ngf: int,
                 nclasses: int=10, nc: int=1) -> None:
        super(SyntheticDataGenerator, self).__init__()
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

    def forward(self, targets: torch.Tensor=None):
        if targets is None:
            targets = self.classrange
            cond = self.classeye
        else:
            cond = self.classeye[targets]
        batch_size = cond.size(0)
        noise = torch.randn(batch_size, self.nz, device=cond.device)
        z = torch.cat((cond, noise), dim=1).unsqueeze(2).unsqueeze(3)
        return self.data_net(z), targets
