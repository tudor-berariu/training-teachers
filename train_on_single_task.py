import os
from argparse import Namespace
from collections import OrderedDict
from tabulate import tabulate
from termcolor import colored as clr
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from liftoff.config import dict_to_namespace

from utils import get_kwargs, get_optimizer, print_nparams
from loss_utils import l2
from models import classifiers
from tasks.datasets import get_loaders


# -- Loading a model or an optimizer

def get_model(module, model_args, *args, **kwargs):
    cfgargs = get_kwargs(model_args)
    print("[MAIN_] The arguments for the", clr(model_args.name, 'red'),
          "model: ", cfgargs)
    return getattr(module, model_args.name)(*args, **cfgargs, **kwargs)


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
        print(clr(f"[TEST_] Avg. loss = {test_loss:.4f} | "
                  f"Acccuracy = {correct:d} / {total:d} "
                  f"({accuracy:.2f}%)", "red"))
    return accuracy

# -- Professor evaluation


def test_professor(agent, device, test_loader, args, state_dict=None):
    all_accs = []  # For all optimizers
    for run_no in range(args.evaluation.nstudents):
        student = get_model(classifiers, args.student,
                            in_size=(1, 32, 32), nclasses=10).to(device)
        if state_dict is not None:
            student.load_state_dict(state_dict)
        student_optimizer = get_optimizer(student.parameters(),
                                          args.student_optimizer)
        start_acc = test(student, device, test_loader, verbose=False)

        print(f"[TEACH] start={start_acc:.2f} ->> ", end="")
        for step in range(args.evaluation.teaching_steps):
            student_optimizer.zero_grad()
            synthetic_loss = agent.eval_student(student)
            if args.c_l2 > 0:
                l2_loss = l2(student.parameters()) * args.c_l2
                synthetic_loss += l2_loss
            synthetic_loss.backward()
            student_optimizer.step()
            if (step + 1) % args.evaluation.teaching_eval_freq == 0:
                acc = test(student, device, test_loader, verbose=False)
                acc -= start_acc
                student.train()
                all_accs.append(acc)
                print(f" {acc:.1f}%", end="")
        print(" !")

    final_score = np.mean(all_accs)
    print(clr(f"[TEACH_] Final score = {final_score:.3f}", "yellow"))

    return final_score


def run(args: Namespace):

    # -- Initialize device and random number generator
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.run_id)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[MAIN_] Using device {device}.")

    # -- Task configuration
    train_loader, test_loader = get_loaders(args.batch_size,
                                            args.test_batch_size,
                                            use_cuda=use_cuda,
                                            in_size=tuple(args.in_size))
    train_loader.to(device)
    test_loader.to(device)

    print("[MAIN_] We haz data loaders.")

    # -- Student model and optimizer

    students, student_optimizers = [], []
    for _idx in range(args.nstudents):
        # TODO: change nin and nout when changing datasets
        student = get_model(classifiers, args.student,
                            in_size=(1, 32, 32), nclasses=10).to(device)
        student_optimizer = get_optimizer(student.parameters(),
                                          args.student_optimizer)
        students.append(student)
        student_optimizers.append(student_optimizer)
        print_nparams(student, name="Student model")

    print("[MAIN_] Initialized", len(students), "students.")

    if not args.evaluation.random_params:
        state_dict = OrderedDict([(name, t.clone()) for (name, t)
                                  in students[0].state_dict().items()])
        print("[MAIN_] Saved initial parameters of student 0.")

    # -- Loss learning

    llargs = args.loss_learning
    if not hasattr(llargs, "debug"):
        llargs.debug = args.debug
    llargs.c_l2 = args.c_l2
    llargs.in_size = (1, 32, 32)
    llargs.nclasses = 10

    if args.professor_type == "generative":
        from agents.generative_agent import GenerativeAgent
        agent = GenerativeAgent(students, llargs)
        agent.to(device)
    else:
        agent = None
        raise NotImplementedError

    for student in students:
        student.train()

    seen_examples = 0

    scores = []
    last_seen, last_student_eval, last_professor_eval = 0, 0, 0
    found_nan = False

    student_trace = []
    all_students_trace = []
    professor_trace = OrderedDict({})

    reset_students = args.nstudents > 1 and args.reset_student > 0

    for epoch in range(args.epochs_no):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            for student_optimizer in student_optimizers:
                student_optimizer.zero_grad()

            student_losses, professor_losses = agent.process(data, target)

            if student_losses is None:
                found_nan = True
                break

            if args.debug:
                pg_ratio = []
                for p in students[0].parameters():
                    pg_ratio.append(f"{(p.data / (p.grad.data + 1e-9)).abs().mean().item():.2f}")
                print(pg_ratio)

            for student_optimizer in student_optimizers:
                student_optimizer.step()

            student_trace.append(student_losses[0])
            all_students_trace.extend(student_losses)
            for name, value in professor_losses.items():
                professor_trace.setdefault(name, []).append(value)

            seen_examples += len(data)

            if reset_students:
                p_reset = len(data) / args.reset_student
                if np.random.sample() < p_reset:
                    idx = np.random.randint(1, args.nstudents)
                    print("[MAIN_] Reset student", idx)
                    students[idx].reset_weights()
                    new_optimizer = get_optimizer(students[idx].parameters(),
                                                  args.student_optimizer)
                    student_optimizers[idx] = new_optimizer

            if seen_examples - last_seen >= args.log_interval:
                details = [("Epoch", epoch + 1),
                           ("Progress (%)", 100. * (batch_idx + 1) / len(train_loader)),
                           ("Student 0", np.mean(student_trace)),
                           ("All students", np.mean(all_students_trace))]
                details.extend([(n, np.mean(vals)) for (n, vals) in professor_trace.items()])
                print(tabulate(details))
                student_trace.clear()
                all_students_trace.clear()
                professor_trace.clear()
                last_seen += args.log_interval

            if seen_examples - last_student_eval >= args.student_eval_interval:
                test(students[0], device, test_loader)
                students[0].train()
                last_student_eval += args.student_eval_interval

            if seen_examples - last_professor_eval >= args.professor_eval_interval:
                if not args.evaluation.random_params:
                    start_params = OrderedDict([(name, t.clone().detach()) for (name, t)
                                                in state_dict.items()])
                else:
                    start_params = None

                score = test_professor(agent, device, test_loader, args,
                                       state_dict=start_params)
                print(f"[*****] Fitness = {score:.2f}")
                scores.append(score)
                last_professor_eval += args.professor_eval_interval

        if found_nan:
            print("Found NaN.")
            break

    if not found_nan and seen_examples > last_student_eval:
        test(students[0], device, test_loader)
        students[0].train()

    if not found_nan and seen_examples > last_professor_eval:
        if not args.evaluation.random_params:
            start_params = OrderedDict([(name, t.clone().detach()) for (name, t)
                                        in state_dict.items()])
        else:
            start_params = None

        score = test_professor(agent, device, test_loader, args,
                               state_dict=start_params)
        print(f"[*****] Fitness = {score:.2f}")
        scores.append(score)

    if found_nan:
        scores = [-1.0]

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
