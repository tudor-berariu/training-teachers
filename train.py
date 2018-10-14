import os
import os.path
from argparse import Namespace
from collections import OrderedDict
from termcolor import colored as clr
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from liftoff.config import dict_to_namespace

from utils import get_kwargs, l2
import models.classifiers as classifiers
from models.loss_predictors import LossLearning

from tasks.datasets import get_loaders


# -- Loading a model or an optimizer

def get_model(module, model_args, *args, **kwargs):
    cfgargs = get_kwargs(model_args)
    print("[MAIN_] The arguments for the", clr(model_args.name, 'red'),
          "model: ", cfgargs)
    return getattr(module, model_args.name)(*args, **cfgargs, **kwargs)


def get_optimizer(parameters, opt_args: Namespace) -> optim.Optimizer:
    cfgargs = get_kwargs(opt_args)
    print("[MAIN_] The arguments for the", clr(opt_args.name, 'red'),
          "optimizer: ", cfgargs)
    return getattr(optim, opt_args.name)(parameters, **cfgargs)


# -- Student evaluation


def test(model, device, test_loader, verbose=True):
    model.eval()
    test_loss = .0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item() * len(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)

    test_loss /= total
    accuracy = (100. * correct / total)
    if verbose:
        print(f"[TEST_] Avg. loss = {test_loss:.4f} | "
              f"Acccuracy = {correct:d} / {total:d} "
              f"({accuracy:.2f}%)")
    return accuracy

# -- Professor evaluation


def test_professor(professor, device, test_loader, args, state_dict=None):
    all_accs = []  # For all optimizers
    for optimizer_args in args.student_optimizers:
        if isinstance(optimizer_args, dict):
            optimizer_args = dict_to_namespace(optimizer_args)
        # TODO: changed the fixed @nin and @nout below
        task_model = get_model(classifiers, args.task_model, nin=1024, nout=10).to(device)
        if state_dict is not None:
            task_model.load_state_dict(state_dict)
        start_acc = test(task_model, device, test_loader, verbose=False)
        student_optimizer = get_optimizer(task_model.parameters(), optimizer_args)

        max_acc, accs = 0, []
        for step in range(args.evaluation.teaching_steps):
            student_optimizer.zero_grad()
            synthetic_loss = professor.eval_student(task_model)
            if args.l2:
                l2_loss = l2(task_model.named_parameters()) * args.l2
                synthetic_loss += l2_loss
            synthetic_loss.backward()
            student_optimizer.step()
            if (step + 1) % args.evaluation.teaching_eval_freq == 0:
                acc = test(task_model, device, test_loader, verbose=False)
                acc -= start_acc
                task_model.train()
                max_acc = max(acc, max_acc)
                accs.append(acc)
                print(f"[TEACH] [Step={(step + 1):d}] "
                      f"Avg. acc: {np.mean(accs):.2f}% | Max. acc: {max_acc:.2f}%")
        opt_acc = np.mean(accs)
        all_accs.append(opt_acc)

    return max(0, np.mean(all_accs))


def print_nparams(model: nn.Module, name: str="Model") -> None:
    nparams = sum(p.nelement() for p in model.parameters())
    print(f"[MAIN_] {name:s} has {clr(f'{nparams:d}', 'yellow'):s} params.")


# -- Training

def run(args: Namespace):

    # -- Initialize device and random number generator
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.run_id)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[MAIN_] Using device {device}.")

    # -- Task configuration
    train_loader, test_loader = get_loaders(args.batch_size, args.test_batch_size,
                                            use_cuda=use_cuda, in_size=(1, 32, 32))
    train_loader.to(device)
    test_loader.to(device)

    # -- Task model and optimizer
    task_model = get_model(classifiers, args.task_model, nin=1024, nout=10).to(device)
    task_optimizer = get_optimizer(task_model.parameters(), args.task_optimizer)
    print_nparams(task_model, name="Task model")

    if not args.evaluation.random_params:
        state_dict = OrderedDict([(name, t.clone()) for (name, t)
                                  in task_model.state_dict().items()])

    # -- Loss learning

    llargs = args.loss_learning
    loss_learner = LossLearning(task_model, llargs.loss_predictor, llargs.nprofessors)
    loss_learner = loss_learner.to(device)
    print_nparams(loss_learner, name="Loss learner")

    loss_optimizer = get_optimizer(loss_learner.professor_parameters(), llargs.optimizer)

    nprofessors = loss_learner.nprofessors
    nsamples = llargs.param_samples

    task_model.train()
    loss_learner.train()

    task_losses = []
    ll_losses = []

    seen_examples = 0
    if hasattr(train_loader, "dataset"):
        total_examples = len(train_loader.dataset)
    else:
        total_examples = train_loader.length

    scores = []

    last_seen, last_eval = 0, 0

    found_nan = False

    if args.professor_type == "parameter":
        from teacher_training import get_batch_processor
        process_batch = get_batch_processor(task_model, loss_learner, args, llargs)
    elif args.professor_type == "generator":
        from generators_training import get_batch_processor
        process_batch = get_batch_processor(task_model, loss_learner, args, llargs)
    else:
        raise ValueError(args.professor_type)

    for epoch in range(args.epochs_no):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            task_optimizer.zero_grad()
            loss_optimizer.zero_grad()

            task_loss, professor_loss, found_nan = process_batch(data, target)

            if found_nan:
                break

            ll_losses.append(professor_loss)
            task_losses.append(task_loss)

            loss_optimizer.step()
            task_optimizer.step()

            seen_examples += len(data)

            if seen_examples - last_seen >= args.log_interval:
                print(f"[Train] #{(epoch + 1):d} "
                      f"[{seen_examples:d}/{total_examples:d}"
                      f"({100. * (batch_idx + 1) / len(train_loader):.0f}%)]"
                      f"\tTask loss: {np.mean(task_losses):.6f}"
                      f"\tLoss loss: {np.mean(ll_losses):.6f}")
                task_losses.clear()
                ll_losses.clear()
                last_seen += args.log_interval

            if seen_examples - last_eval >= args.eval_interval:
                test(task_model, device, test_loader)
                task_model.train()
                if not args.evaluation.random_params:
                    start_params = OrderedDict([(name, t.clone().detach()) for (name, t)
                                                in state_dict.items()])
                else:
                    start_params = None

                score = test_professor(loss_learner, device, test_loader, args, state_dict=start_params)
                print(f"[*****] Fitness = {score:.2f}")
                scores.append(score)
                last_eval += args.eval_interval
        if found_nan:
            break

    if not found_nan and seen_examples > last_eval:
        test(task_model, device, test_loader)
        task_model.train()
        score = test_professor(loss_learner, device, test_loader, args)
        scores.append(score)
    elif found_nan:
        scores = [-1]

    print(f"Final score: {np.mean(scores):.3f}")

    with open(os.path.join(args.out_dir, "fitness"), "w") as handler:
        handler.write(f"{np.mean(scores):f}\n")

    return np.mean(scores)


# -- The usual main function to be used with liftoff

def main() -> None:
    # Reading args
    from time import time
    from liftoff.config import read_config

    args = read_config()  # type: Args

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
