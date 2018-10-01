import os
import os.path
from typing import Iterator, Tuple
from argparse import Namespace
from collections import OrderedDict
from itertools import chain
from termcolor import colored as clr
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torchvision import datasets, transforms

from liftoff.config import dict_to_namespace

from utils import get_kwargs
import models.classifiers as classifiers
from models.loss_predictors import LossLearning


# -- The problem to be solved

def get_loaders(args: Namespace, use_cuda: bool):
    # TODO: Load other datasets
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./.data/fashion',
                              train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.2860,), (0.3530,))
                              ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./.data/fashion',
                              train=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.2860,), (0.3530,))
                              ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader


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


# -- Some cost functions that are applied on iterable collections of tensors

def collapse(parameters: Iterator[Tensor], detach: bool= False):
    # TODO: collapse batches of parameters
    if detach:
        return torch.cat(tuple(t.clone().view(-1).detach() for t in parameters))

    return torch.cat(tuple(t.view(-1) for t in parameters))


def l2(named_parameters: Iterator[Tuple[str, Tensor]]) -> Tensor:
    return sum(param.pow(2).sum() for (_, param) in named_parameters)


def cos(parameters: Iterator[Tensor], targets: Iterator[Tensor]):
    flat_parameters = collapse(parameters).unsqueeze(0)
    flat_targets = collapse(targets, detach=True).unsqueeze(0)
    return F.cosine_embedding_loss(flat_parameters, flat_targets,
                                   torch.Tensor([1]).to(flat_parameters.device))


def mse(params: Iterator[Tensor], targets: Iterator[Tensor]) -> Tensor:
    mse_loss = nn.MSELoss(reduction='sum')
    return sum(mse_loss(y, t.detach()) for (y, t) in zip(params, targets))

# -- Student evaluation


def test(model, device, test_loader, verbose=True):
    model.eval()
    test_loss = .0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item() * len(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = (100. * correct / len(test_loader.dataset))
    if verbose:
        print(f"[TEST_] Avg. loss = {test_loss:.4f} | "
              f"Acccuracy = {correct:d} / {len(test_loader.dataset)} "
              f"({accuracy:.2f}%)")
    return accuracy

# -- Professor evaluation


def test_professor(professor, device, test_loader, args, state_dict=None):
    all_accs = []  # For all optimizers
    for optimizer_args in args.student_optimizers:
        if isinstance(optimizer_args, dict):
            optimizer_args = dict_to_namespace(optimizer_args)
        # TODO: changed the fixed @nin and @nout below
        task_model = get_model(classifiers, args.task_model, nin=784, nout=10).to(device)
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
    train_loader, test_loader = get_loaders(args, use_cuda)

    # -- Task model and optimizer
    task_model = get_model(classifiers, args.task_model, nin=784, nout=10).to(device)
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
    total_examples = len(train_loader.dataset)

    scores = []

    last_seen, last_eval = 0, 0

    ag_kwargs = {"create_graph": True, "retain_graph": True, "only_inputs": True}

    found_nan = False

    for epoch in range(args.epochs_no):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            task_optimizer.zero_grad()
            loss_optimizer.zero_grad()

            # -- Compute noisy variants of the current parameters of the task model

            batched_params = OrderedDict({})  # type: OrderedDict[str, List[Tensor]]
            used_params = [OrderedDict({}) for _ in range(nsamples)]
            noisy_params = [OrderedDict({}) for _ in range(nsamples)]

            for name, param in task_model.named_parameters():
                lst = []
                for np_idx in range(nsamples):
                    noise = torch.randn_like(param) * llargs.noise_intensity
                    if llargs.noise_type == "absolute":
                        perturbation = noise
                    elif llargs.noise_type == "relative":
                        perturbation = noise * param.detach()
                    elif llargs.noise_type == "max_layer":
                        perturbation = noise * param.max().detach().item()
                    else:
                        raise ValueError
                    perturbation.detach_()
                    noisy_params[np_idx][name] = noisyp = param + perturbation

                    if llargs.full_grad:
                        used_params[np_idx][name] = noisyp
                        lst.append(noisyp)
                    else:
                        cloned_noisyp = noisyp.clone().detach_()
                        cloned_noisyp.requires_grad = True
                        used_params[np_idx][name] = cloned_noisyp
                        lst.append(cloned_noisyp)
                batched_params[name] = torch.stack(tuple(lst))

            # ^^ The used_params are either task_models's params or some clones

            # -- Compute the loss predictions

            loss_predictions = loss_learner(batched_params)
            # ^^ loss_predictions is a nprofessors -length list of tensors of size (nsamples,)

            outputs = [task_model(data, noisyp) for noisyp in noisy_params]
            # ^^ outputs is a list of length `nsamples` with tensors of size len(data)

            # -- Compute loss for each set of parameters

            nlls = [F.cross_entropy(y, target) for y in outputs]
            stacked_nlls = torch.stack(nlls)
            avg_nll = stacked_nlls.mean()

            if args.l2:
                task_loss = avg_nll + l2(task_model.named_parameters()) * args.l2
            else:
                task_loss = avg_nll

            task_losses.append(task_loss.item())

            # -- Compute loss for each professor

            # INFO: torch.stack has some issues with autograd, we avoid it for now

            loss1, loss2 = 0, 0
            losscos = 0
            lossoptim = 0

            loss0 = F.mse_loss(torch.stack(loss_predictions),
                               stacked_nlls.unsqueeze(0).expand(nprofessors, nsamples).detach())

            if llargs.sobolev_depth > 0:
                target_g = autograd.grad(nlls, chain(*(p.values() for p in noisy_params)),
                                         **ag_kwargs)
                if llargs.sobolev_depth > 1:
                    rand_v = [torch.bernoulli(torch.rand_like(g)) for g in target_g]
                    target_Hv = autograd.grad(target_g, chain(*(p.values() for p in noisy_params)),
                                              grad_outputs=rand_v, **ag_kwargs)

                for k in range(nprofessors):
                    g_k = autograd.grad(loss_predictions[k],
                                        chain(*(p.values() for p in used_params)),
                                        grad_outputs=torch.ones_like(loss_predictions[k]),
                                        **ag_kwargs)
                    loss1 += mse(g_k, target_g)
                    losscos += cos(g_k, target_g)

                    if llargs.sobolev_depth > 1:
                        Hv_k = autograd.grad(g_k,
                                             chain(*(p.values() for p in used_params)),
                                             grad_outputs=rand_v,
                                             **ag_kwargs)
                        loss2 += mse(Hv_k, target_Hv)

                    if llargs.optim_loss:
                        g_k_iter = iter(g_k)
                        for usedp in used_params:
                            one_step_params = OrderedDict({})
                            for key, value in usedp.items():
                                one_step_params[key] = value - .01 * next(g_k_iter)
                            next_output = task_model(data, one_step_params)
                            next_nll = F.cross_entropy(next_output, target)
                            lossoptim += next_nll

            professor_loss = llargs.c0 * loss0 + \
                llargs.c1 * loss1 + \
                llargs.ccos * losscos + \
                llargs.c2 * loss2 + \
                llargs.coptim * lossoptim

            if torch.isnan(professor_loss).any().item():
                found_nan = True
                break

            ll_losses.append(professor_loss.item())

            # -- Perform backpropagation

            avg_nll.backward(retain_graph=llargs.full_grad)
            professor_loss.backward()

            loss_optimizer.step()
            task_optimizer.step()

            seen_examples += len(data)

            if args.debug:
                print("[DEBUG]", loss0.item(), loss1.item(), loss2.item())
                # print([p.grad.abs().max().item() for p in loss_learner.parameters()])

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
        scores = [.0]

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
