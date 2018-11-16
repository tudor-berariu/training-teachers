from argparse import Namespace
import os
import torch
import torch.nn.functional as F
import numpy as np
from termcolor import colored as clr
from models import classifiers
from models import get_model
from utils import printer, args_to_dict
from utils import get_optimizer
from tasks.datasets import get_loaders
from train_on_single_task import test


def run(args: Namespace) -> float:

    assert __file__.endswith(args.script)

    if args.wandb:
        import wandb
        wandb.init()
        wandb.config.update(args_to_dict(args, sep="_"))

    verbose = int(args.verbose) if hasattr(args, "verbose") else 1
    info = printer("MAIN", 1, verbose=verbose)

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
    # Prepare data loaders.

    train_loader, test_loader, data_info = \
        get_loaders(args.dataset, args.batch_size,
                    args.test_batch_size, in_size=tuple(args.in_size))
    train_loader.to(device)
    test_loader.to(device)

    in_size, nclasses, nrmlz = data_info
    args.in_size = in_size
    args.nclasses = nclasses

    info("Prepared loaders for", clr(args.dataset, "red"),
         "with", nclasses, "classes.")


    # -------------------------------------------------------------------------
    #
    # Start training

    student = get_model(classifiers, args.student, in_size=args.in_size,
                        nclasses=args.nclasses).to(device)

    student_optimizer = get_optimizer(student.parameters(),
                                      args.student_optimizer)

    accs = []
    for epoch in range(args.nepochs):
        student.train()
        for data, target in train_loader:
            output = student(data)
            loss = F.cross_entropy(output, target)
            student_optimizer.zero_grad()
            loss.backward()
            student_optimizer.step()
        acc, _, _ = test(student, test_loader, device)
        accs.append(acc)
        wandb.log({"acc": acc})

    fitness = np.mean(accs[:5])

    info(f"Final fitness: {fitness:.3f}")

    with open(os.path.join(args.out_dir, "fitness"), "w") as handler:
        handler.write(f"{fitness:f}\n")

    if args.wandb:
        wandb.run.summary["first_five"] = np.mean(accs[:5])
        wandb.run.summary["last_five"] = np.mean(accs[5:])
        wandb.run.summary["avg_acc"] = np.mean(accs)

    return fitness


if __name__ == "__main__":
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