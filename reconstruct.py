from argparse import Namespace
from functools import reduce
import pickle
from operator import mul
import os
from typing import List, Tuple, Union
import numpy as np
import torch
from torch import Tensor as T
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid

from models import get_model
from models import classifiers
from utils import printer, args_to_dict
from utils import get_optimizer
from tasks.datasets import get_loaders
from train_on_single_task import test_professor


def best_and_last(values, wlen) -> Tuple[float, float]:
    avgs = np.convolve(values, np.ones((wlen,)) / wlen)[(wlen - 1):(1 - wlen)]
    return avgs.max(), avgs[-1]


def entropy(scores: T) -> T:
    exp = scores.exp()
    sum_exp = exp.sum(dim=1)
    return -((scores * exp).sum(dim=1) / sum_exp - sum_exp.log()).mean().item()


class DataModel(nn.Module):

    def __init__(self, nproto, nreal: int,
                 size: List[int], nclasses: int) -> None:
        super(DataModel, self).__init__()
        self.size = size
        self.nclasses = nclasses
        self.nproto = nproto
        self.nreal = nreal
        self.proto_data = nn.Parameter(torch.rand(nproto, reduce(mul, size)))
        self.proto_targets = nn.Parameter(torch.randn(nproto, nclasses))
        self.attention = nn.Parameter(torch.randn(nreal, nproto))

        self._eval_on_proto = False
        self._eval_batch_size = 32
        self._real_targets = None

    def forward(self, *args, **kwargs):
        # pylint: disable=W0221
        raise NotImplementedError

    def sample_fake(self,
                    nsamples: int = None,
                    idxs: T = None,
                    return_targets: bool = True,
                    real_targets: bool = False) -> Union[T, Tuple[T, T]]:
        attention = self.attention
        if nsamples is not None:
            idxs = torch.randint(0, self.nproto, (nsamples,),
                                 device=attention.device, dtype=torch.long)
        if idxs is not None:
            attention = attention.index_select(0, idxs)
        attention = F.softmax(attention, dim=1)

        proto_data = torch.sigmoid(self.proto_data)
        fake_data = (attention @ proto_data).view(-1, *self.size)
        if not return_targets:
            return fake_data
        if not real_targets:
            proto_targets = torch.softmax(self.proto_targets, dim=1)
            fake_targets = attention.detach() @ proto_targets
            return fake_data, fake_targets
        if idxs is None:
            return fake_data, self._real_targets
        return fake_data, self._real_targets[idxs]

    def sample_proto(self, nsamples: int = 1) -> Union[T, Tuple[T, T]]:
        idxs = torch.randint(0, self.nproto, (nsamples,),
                             device=self.proto_data.device,
                             dtype=torch.long)
        proto_data = torch.sigmoid(self.proto_data[idxs]).view(-1, *self.size)
        proto_targets = torch.softmax(self.proto_targets[idxs], dim=1)
        return proto_data, proto_targets

    def attention_entropy(self):
        with torch.no_grad():
            return entropy(self.attention)

    def proto_targets_entropy(self):
        with torch.no_grad():
            return entropy(self.proto_targets)

    @property
    def eval_on_proto(self) -> bool:
        return self._eval_on_proto

    @eval_on_proto.setter
    def eval_on_proto(self, value: bool) -> None:
        self._eval_on_proto = bool(value)

    def use_real_targets(self, value: bool = False, targets: T = None) -> None:
        if value:
            if targets.size(0) != self.nreal:
                raise ValueError(f"Expected a tensor of size {self.nreal}.")
            self._real_targets = targets
        else:
            self._real_targets = None

    def eval_student(self, student: nn.Module, _step: int) -> Tuple[T, float]:
        batch_size = self._eval_batch_size
        with torch.no_grad():
            if self._eval_on_proto:
                data, targets = self.sample_proto(nsamples=batch_size)
            else:
                data, targets = self.sample_fake(nsamples=batch_size)
        output = student(data)
        if targets.ndimension() == 2:
            loss = F.kl_div(F.log_softmax(output, dim=1),
                            F.softmax(targets, dim=1))
            correct = output.max(dim=1)[1].eq(targets.max(dim=1)[1]).sum()
        else:
            loss = F.cross_entropy(output, targets)
            correct = output.max(dim=1)[1].eq(targets).sum()

        return loss, 100 * float(correct) / len(targets)


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

    train_loader, test_loader, data_info = \
        get_loaders(args.dataset, args.nreal,
                    args.test_batch_size, in_size=tuple(args.in_size),
                    normalize=False, limit=args.nreal)
    train_loader.to(device)
    test_loader.to(device)

    real_data, real_targets, _ = next(iter(train_loader))
    del train_loader

    nreal, *size = real_data.size()
    args.nclasses = nclasses = data_info[1]

    ridxs = torch.randperm(nreal)[:72].long().to(device)

    # -------------------------------------------------------------------------
    #
    # Initialize data model and its optimizer.

    data_model = DataModel(args.nfake, nreal, size, nclasses).to(device)
    optimizer = get_optimizer(data_model.parameters(), args.optimizer)

    # -------------------------------------------------------------------------
    #
    # Initialize student.

    students = []
    for _ in range(args.students_per_batch):
        students.append(get_model(classifiers, args.student,
                                  in_size=size, nclasses=nclasses).to(device))

    # -------------------------------------------------------------------------
    #
    # Start training

    fake_rt_accs, fake_ft_accs, proto_accs = [], [], []

    for step in range(1, args.steps_no + 1):

        data_model.zero_grad()
        idxs = torch.randint(0, nreal, (args.batch_size,),
                             dtype=torch.long, device=device)
        fake_data, fake_targets = data_model.sample_fake(idxs=idxs)
        crt_data, crt_targets = real_data[idxs], real_targets[idxs]
        target_nll = F.cross_entropy(fake_targets, crt_targets)

        with torch.no_grad():
            mse = F.mse_loss(fake_data, crt_data).item()

        data_kls = []
        data_kl = 0

        data_model.attention_entropy()

        for student in students:
            with torch.no_grad():
                real_probs = F.softmax(student(crt_data), dim=1)

            fake_output = student(fake_data)
            fake_logp = F.log_softmax(fake_output, dim=1)
            crt_kl = F.kl_div(fake_logp, real_probs)
            data_kl += crt_kl
            data_kls.append(crt_kl.item())

        loss = target_nll + (data_kl / args.students_per_batch)
        loss.backward()
        optimizer.step()

        if step % args.reset_freq == 0:
            for student in students:
                student.reset_weights()

        avg_kl = np.mean(data_kls)

        if args.tensorboard:
            writer.add_scalar('data-KL', avg_kl, step)
            writer.add_scalar('target-NLL', target_nll.item(), step)
            writer.add_scalar('data-MSE', mse, step)
            writer.add_scalar('attn-entropy',
                              data_model.attention_entropy(), step)
            writer.add_scalar('proto-targets-entropy',
                              data_model.proto_targets_entropy(), step)

        info(f"Step {step:5d} | KL = {avg_kl:10.4f} | MSE = {mse:10.4f} | "
             f"target NLL =  {target_nll.item():10.4f}")

        if step % args.eval_freq == 0:
            data_model.eval_on_proto = False
            data_model.use_real_targets(True, real_targets)
            fake_acc_1 = test_professor(data_model, test_loader, device, args)

            data_model.use_real_targets(False)
            fake_acc_2 = test_professor(data_model, test_loader, device, args)

            data_model.eval_on_proto = True
            proto_acc = test_professor(data_model, test_loader, device, args)

            writer.add_scalar('teach/faked-realt-acc', fake_acc_1, step)
            writer.add_scalar('teach/faked-faket-acc', fake_acc_2, step)
            writer.add_scalar('teach/proto-acc', proto_acc, step)

            fake_rt_accs.append(fake_acc_1)
            fake_ft_accs.append(fake_acc_2)
            proto_accs.append(proto_acc)

            info(f"Step {step:5d} | Fake (RT): {fake_acc_1:6.2f}% | "
                 f"Fake (FT): {fake_acc_2:6.2f}% | Proto: {proto_acc:6.2f}%")

        if step % args.report_freq == 0:
            with torch.no_grad():
                proto_samples, _ = data_model.sample_proto(nsamples=144)
            proto_path = os.path.join(args.out_dir,
                                      f"proto_samples_{step:04d}.png")
            save_image(proto_samples, proto_path, nrow=12)

            with torch.no_grad():
                rec = data_model.sample_fake(idxs=ridxs, return_targets=False)
                both = torch.cat((real_data[ridxs], rec), dim=0)
            fake_path = os.path.join(args.out_dir,
                                     f"fake_samples_{step:04d}.png")
            save_image(both, fake_path, nrow=12)

            if args.tensorboard:
                proto_grid = make_grid(proto_samples, nrow=12)
                writer.add_image('Proto', proto_grid, step)
                compare_grid = make_grid(both, nrow=12)
                writer.add_image('Reconstructions', compare_grid, step)

    if args.tensorboard:
        writer.export_scalars_to_json(os.path.join(args.out_dir, "trace.json"))
        writer.close()

    best_acc1, last_acc1 = best_and_last(fake_rt_accs, 5)
    best_acc2, last_acc2 = best_and_last(fake_ft_accs, 5)
    best_acc3, last_acc3 = best_and_last(proto_accs, 5)

    summary = {
        "best_fake_rt": best_acc1, "last_fake_rt": last_acc1,
        "best_fake_ft": best_acc2, "last_fake_ft": last_acc2,
        "best_proto": best_acc3, "last_proto": last_acc3,
    }

    with open(os.path.join(args.out_dir, 'summary.pkl'), 'wb') as handler:
        pickle.dump(summary, handler, pickle.HIGHEST_PROTOCOL)


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
