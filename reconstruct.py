from argparse import Namespace
from typing import List, Tuple
from functools import reduce
from operator import mul
import os
import numpy as np

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from models import get_model
from models import classifiers
from utils import printer, args_to_dict
from utils import get_optimizer


class DataModel(nn.Module):

    def __init__(self, nproto, nreal: int,
                 size: List[int], nclasses: int) -> None:
        super(DataModel, self).__init__()
        nfeatures = reduce(mul, size)
        self.size = size
        self.nclasses = nclasses
        self.nproto = nproto
        self.nreal = nreal
        self.proto_data = nn.Parameter(torch.rand(nproto, nfeatures))
        self.proto_targets = nn.Parameter(torch.randn(nproto, nclasses))
        self.attention = nn.Parameter(torch.randn(nreal, nproto))

    def forward(self, idxs: Tensor = None) -> Tuple[Tensor, Tensor]:
        # pylint: disable=W0221
        attention = self.attention
        if idxs is not None:
            attention = attention.index_select(0, idxs)
        attention = F.softmax(attention, dim=1)

        proto_data = torch.sigmoid(self.proto_data)
        fake_data = (attention @ proto_data).view(-1, *self.size)
        proto_targets = torch.softmax(self.proto_targets, dim=1)
        fake_targets = attention.detach() @ proto_targets
        return fake_data, fake_targets

    def sample_proto(self, nsamples: int = 1) -> Tensor:
        idxs = torch.randint(0, self.nproto, (nsamples,),
                             device=self.proto_data.device,
                             dtype=torch.long)
        proto_data = torch.sigmoid(self.proto_data[idxs]).view(-1, *self.size)
        proto_targets = torch.softmax(self.proto_targets[idxs], dim=1)
        return proto_data, proto_targets

    def sample_fake(self, nsamples: int = 1) -> Tensor:
        idxs = torch.randint(0, self.nreal, (nsamples,),
                             device=self.attention.device,
                             dtype=torch.long)
        fake_data, fake_targets = self.forward(idxs=idxs)
        return fake_data, F.softmax(fake_targets, dim=1)


def get_data(args: Namespace) -> torch.Tensor:
    trainset = getattr(datasets, args.dataset)(
        f'./.data/{args.dataset:s}',
        train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = DataLoader(trainset, batch_size=args.nreal, shuffle=True)
    data, target = next(iter(train_loader))
    return data, target


def run(args: Namespace):

    # -------------------------------------------------------------------------
    #
    # Initialize loggers, tensorboard, wandb

    verbose = int(args.verbose) if hasattr(args, "verbose") else 1
    info = printer("MAIN", 1, verbose=verbose)

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir=args.out_dir)

    if args.wandb:
        import wandb
        wandb.init()
        wandb.config.update(args_to_dict(args))

    # -------------------------------------------------------------------------
    #
    # Initialize device and random number generator.

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if hasattr(args, "run_id") and int(args.run_id) >= 0:
        torch.manual_seed(int(args.run_id))
        info("Seed:", f"{int(args.run_id):d}.")
    device = torch.device("cuda" if use_cuda else "cpu")
    info(f"Using device {device}.")

    # -------------------------------------------------------------------------
    #
    # Collect real data

    real_data, real_targets = get_data(args)
    real_data, real_targets = real_data.to(device), real_targets.to(device)
    nreal, *size = real_data.size()
    nclasses = real_targets.max()

    # -------------------------------------------------------------------------
    #
    # Initialize data model and its optimizer.

    data_model = DataModel(args.nfake, nreal, size, nclasses).to(device)
    optimizer = get_optimizer(data_model.parameters(), args.optimizer)

    # -------------------------------------------------------------------------
    #
    # Initialize students.

    nstudents = args.nstudents
    students = []
    for _idx in range(nstudents):
        student = get_model(classifiers, args.student,
                            in_size=size, nclasses=nclasses)
        students.append(student.to(device))

    # -------------------------------------------------------------------------
    #
    # Start training

    dirty = [True] * nstudents
    real_ps = [None] * nstudents

    for step in range(1, args.steps_no):
        crt_kls, crt_mse = [], 0
        data_model.zero_grad()
        if args.batch_size == 0:
            fake_data, fake_targets = data_model()
            crt_mse = F.mse_loss(fake_data, real_data).item()
        for sidx, student in enumerate(students):
            if args.batch_size == 0:
                if dirty[sidx]:
                    with torch.no_grad():
                        real_ps[sidx] = F.softmax(student(real_data), dim=1)
                    dirty[sidx] = False

            else:
                idx = torch.randint(0, nreal, (args.batch_size,),
                                    dtype=torch.long, device=device)
                with torch.no_grad():
                    real_ps[sidx] = F.softmax(student(real_data[idx]), dim=1)
                fake_data, fake_targets = data_model(idx)
                with torch.no_grad():
                    crt_mse += F.mse_loss(fake_data, real_data[idx]).item()

            fake_output = student(fake_data)
            fake_logp = F.log_softmax(fake_output, dim=1)
            kldiv = F.kl_div(fake_logp, real_ps[sidx])
            (kldiv / nstudents).backward(retain_graph=True)
            crt_kls.append(kldiv.item())

        optimizer.step()
        avg_kl = np.mean(crt_kls)

        if step % args.reset_freq == 0:
            for sidx, student in enumerate(students):
                dirty[sidx] = True
                student.reset_weights()

        if args.tensorboard:
            writer.add_scalar('KL', avg_kl, step)
            writer.add_scalar('MSE', crt_mse, step)
        info(f"Step {step:5d} | KL = {avg_kl:10.4f} | MSE = {crt_mse:10.4f}")

        if step % args.report_freq == 0:
            with torch.no_grad():
                proto_samples, _ = data_model.sample_proto(nsamples=64)

                fake_samples, _ = data_model.sample_fake(nsamples=100)
            save_image(proto_samples,
                       os.path.join(args.out_dir,
                                    f"proto_samples_{step:04d}.png"))
            idxs = torch.randint(0, nreal, (32,),
                                 device=device, dtype=torch.long)
            with torch.no_grad():
                fake_samples, _ = data_model.forward(idxs)
                both = torch.cat((real_data[idxs], fake_samples), dim=0)

            save_image(both,
                       os.path.join(args.out_dir,
                                    f"fake_samples_{step:04d}.png"))

            if args.tensorboard:
                proto_grid = make_grid(proto_samples)
                writer.add_image('Proto', proto_grid, step)
                compare_grid = make_grid(both)
                writer.add_image('Reconstructions', compare_grid, step)

    if args.tensorboard:
        writer.export_scalars_to_json(os.path.join(args.out_dir, "trace.json"))
        writer.close()

def main() -> None:
    # Reading args
    from time import time
    from liftoff.config import read_config

    args = read_config()  # type: Namespace

    if not hasattr(args, "out_dir"):
        if not os.path.isdir('./results'):
            os.mkdir('./results')
        out_dir = f'./results/{str(int(time())):s}_{args.experiment:s}'
        os.mkdir(out_dir)
        args.out_dir = out_dir
    elif not os.path.isdir(args.out_dir):
        raise Exception(f"Directory {args.out_dir} does not exist.")

    if not hasattr(args, "run_id"):
        args.run_id = 0

    run(args)


if __name__ == "__main__":
    main()
