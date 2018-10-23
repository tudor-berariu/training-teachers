import os
import pickle
from argparse import Namespace
from collections import OrderedDict
from tabulate import tabulate
from termcolor import colored as clr
import numpy as np

import torch
from torch import Tensor
import torch.nn.functional as F

from utils import get_optimizer, print_nparams, grad_info, print_conf
from loss_utils import l2
from models import get_model
from models import classifiers
from models.classifiers import sample_classifier
from tasks.datasets import get_loaders


# -----------------------------------------------------------------------------
#
# Test a given student on the test data, returns the accuracy.

def test(model, device, test_loader, verbose=True, do_conf=False):
    model.eval()
    test_loss = .0
    correct = 0
    total = 0
    conf = None
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item() * len(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            if do_conf:
                if conf is None:
                    nclasses = output.size(1)
                    conf = torch.zeros(nclasses, nclasses, device=device)
                conf.put_(target * output.size(1) + pred.squeeze(),
                          torch.ones(target.size(), device=device),
                          accumulate=True)
            total += len(data)

    test_loss /= total
    accuracy = (100. * correct / total)
    if do_conf:
        conf = conf / conf.sum(dim=1, keepdim=True) * 100

    if verbose:
        print(clr(f"[TEST_] Avg. loss = {test_loss:.4f} | "
                  f"Acccuracy = {correct:d} / {total:d} "
                  f"({accuracy:.2f}%)", "red"))
        if do_conf:
            print_conf(conf.cpu())
    return accuracy, conf


# -----------------------------------------------------------------------------
#
# Test a given professor by optimizing a few students from random
# parameters.


def test_professor(agent, device, test_loader, args, state_dict=None):
    all_accs = []
    nsteps = args.evaluation.teaching_steps
    for run_no in range(args.evaluation.nstudents):
        print(f"[TEACH] Teaching student {run_no + 1:d}.")
        student = get_model(classifiers, args.student,
                            in_size=args.in_size,
                            nclasses=args.nclasses).to(device)
        print_nparams(student, name=f"Student #{run_no + 1:d}")
        if state_dict is not None:
            student.load_state_dict(state_dict)
        student_optimizer = get_optimizer(student.parameters(),
                                          args.student_optimizer)
        for step in range(nsteps):
            student_optimizer.zero_grad()
            synthetic_loss, synthetic_acc = agent.eval_student(student)
            if args.c_l2 > 0:
                l2_loss = l2(student.parameters()) * args.c_l2
                full_loss = synthetic_loss + l2_loss
            else:
                full_loss = synthetic_loss
            full_loss.backward()
            student_optimizer.step()
            if (step + 1) % args.evaluation.teaching_eval_freq == 0:
                is_last_step = (step + 1) == args.evaluation.teaching_steps
                kwargs = {"verbose": False, "do_conf": is_last_step}
                acc, conf = test(student, device, test_loader, **kwargs)
                student.train()
                all_accs.append(acc)
                l2s = "" if l2_loss is None else f"L2: {l2_loss.item():3f}; "
                print(f"[TEACH][{run_no + 1:d}][{step + 1: 3d}/{nsteps:d}] "
                      f"Synthetic NLL: {synthetic_loss.item():.3f}; " + l2s +
                      f"Synthetic acc: {synthetic_acc:.2f}; " +
                      clr(f"Acc: {acc:.1f}%", "yellow"))
                if is_last_step:
                    print_conf(conf)
    if len(all_accs) > 25:
        all_accs = all_accs[-25:]
    final_score = np.mean(all_accs)
    print(clr(f"[TEACH] >>> Final score = {final_score:.3f} <<<", "yellow"))

    return final_score


def get_n_samples(data_loader, nsamples: int) -> Tensor:
    iter_data = iter(data_loader)
    data, target = None, None
    while data is None or len(data) < nsamples:
        try:
            more_data, more_target = next(iter_data)
        except StopIteration:
            iter_data = iter(data_loader)
            more_data, more_target = next(iter_data)
        end = nsamples - (0 if data is None else len(data))
        more_data, more_target = more_data[:end], more_target[:end]
        if data is None:
            data, target = more_data, more_target
        else:
            data = torch.cat((data, more_data), dim=0)
            target = torch.cat((target, more_target), dim=0)
    return {"data": data.clone(), "target": target.clone()}


def run(args: Namespace):
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements

    # -- Initialize device and random number generator
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.run_id)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[MAIN_] Using device {device}.")

    # -- Task configuration
    train_loader, test_loader, info = get_loaders(args.batch_size,
                                                  args.test_batch_size,
                                                  use_cuda=use_cuda,
                                                  in_size=tuple(args.in_size))
    train_loader.to(device)
    test_loader.to(device)

    some_batch = get_n_samples(train_loader, nsamples=32)

    in_size, nclasses, nrmlz = info
    args.in_size = in_size
    args.nclasses = nclasses

    print("[MAIN_] We haz data loaders.")

    # -- Student model and optimizer

    students, student_optimizers = [], []
    for idx in range(args.nstudents):
        if idx == 0 or not args.random_students:
            student = get_model(classifiers, args.student,
                                in_size=in_size,
                                nclasses=nclasses).to(device)
            if idx > 0 and not args.random_params:
                student.load_state_dict(students[0].state_dict())
        else:
            student = sample_classifier(in_size, nclasses).to(device)
        student_optimizer = get_optimizer(student.parameters(),
                                          args.student_optimizer)
        students.append(student)
        student_optimizers.append(student_optimizer)
        print_nparams(student, name="Student model")

    print("[MAIN_] Initialized", len(students), "students.")

    if not args.random_params:
        state_dict = OrderedDict([(name, t.clone()) for (name, t)
                                  in students[0].state_dict().items()])
        print("[MAIN_] Saved initial parameters of student 0.")
    else:
        state_dict = None
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

    best_fitness, best_epoch, scores, scores_at = None, None, [], []
    last_seen, last_student_eval, last_professor_eval = 0, 0, 0
    found_nan, should_stop = False, False

    student_trace = []
    all_students_trace = []
    professor_trace = OrderedDict({})
    professor_avg_trace = dict({})

    reset_students = args.nstudents > 1 and args.reset_student > 0

    for epoch in range(args.epochs_no):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            for student_optimizer in student_optimizers:
                student_optimizer.zero_grad()

            student_losses, prof_losses = agent.process(data, target, nrmlz)

            if student_losses is None:
                print(clr("[MAIN_] Found NaN. Early stopping.", "red"))
                found_nan = should_stop = True
                break

            if args.debug:
                print("[MAIN_][DEBUG] Student 0:")
                print(tabulate(grad_info(students[0])))

            for student_optimizer in student_optimizers:
                student_optimizer.step()

            student_trace.append(student_losses[0])
            all_students_trace.extend(student_losses)
            for name, value in prof_losses.items():
                professor_trace.setdefault(name, []).append(value)

            seen_examples += len(data)

            if reset_students:
                p_reset = len(data) / args.reset_student
                if np.random.sample() < p_reset:
                    idx = np.random.randint(1, args.nstudents)
                    print("[MAIN_] Reset student", idx)

                    if not args.random_params:
                        student[idx].load_state_dict(state_dict)
                    elif args.random_students:
                        students[idx] = sample_classifier(in_size,
                                                          nclasses).to(device)
                    else:
                        students[idx] = get_model(classifiers, args.student,
                                                  in_size=in_size,
                                                  nclasses=nclasses).to(device)
                    agent.init_student(students[idx], args.student_optimizer)
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
                professor_avg_trace.setdefault("seen", []).append(seen_examples)
                for info, values in professor_trace.items():
                    professor_avg_trace.setdefault(info, []).append(np.mean(values))

                professor_trace.clear()
                last_seen += args.log_interval

            if seen_examples - last_student_eval >= args.student_eval_interval:
                test(students[0], device, test_loader, do_conf=True)
                students[0].train()
                last_student_eval += args.student_eval_interval

            if seen_examples - last_professor_eval >= args.professor_eval_interval:
                if not args.random_params:
                    start_params = OrderedDict([(name, t.clone().detach()) for (name, t)
                                                in state_dict.items()])
                else:
                    start_params = None

                score = test_professor(agent, device, test_loader, args,
                                       state_dict=start_params)
                scores.append(score)
                scores_at.append(seen_examples)
                if len(scores) >= 10:
                    new_avg = np.mean(scores[-10:])
                    if best_fitness is None or new_avg > best_fitness:
                        best_fitness, best_epoch = new_avg, len(scores)

                    if best_epoch + 5 < len(scores):
                        print(clr("[MAIN_] Early stopping.", "red"))
                        should_stop = True
                        break

                last_professor_eval += args.professor_eval_interval

        if should_stop:
            break

        agent.save_state(args.out_dir, epoch, **some_batch)
        with open(os.path.join(args.out_dir, f"trace.th"), 'wb') as handle:
            pickle.dump(professor_avg_trace, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(args.out_dir, f"eval.th"), 'wb') as handle:
            pickle.dump({"scores": scores, "seen": scores_at},
                        handle, protocol=pickle.HIGHEST_PROTOCOL)

    if not found_nan and seen_examples > last_student_eval:
        test(students[0], device, test_loader)
        students[0].train()

    if not found_nan and seen_examples > last_professor_eval:
        if not args.random_params:
            start_params = OrderedDict([(name, t.clone().detach()) for (name, t)
                                        in state_dict.items()])
        else:
            start_params = None

        score = test_professor(agent, device, test_loader, args,
                               state_dict=start_params)
        scores.append(score)
        scores_at.append(seen_examples)
        if len(scores) >= 10:
            new_avg = np.mean(scores[-10:])
            if best_fitness is None or new_avg > best_fitness:
                best_fitness, best_epoch = new_avg, len(scores)

        with open(os.path.join(args.out_dir, f"eval.th"), 'wb') as handle:
            pickle.dump({"scores": scores, "seen": scores_at},
                        handle, protocol=pickle.HIGHEST_PROTOCOL)

    if found_nan:
        fitness = -1.0
    elif best_fitness is None:
        fitness = np.mean(scores)
    else:
        fitness = best_fitness

    print(f"Final fitness: {fitness:.3f}")

    with open(os.path.join(args.out_dir, "fitness"), "w") as handler:
        handler.write(f"{fitness:f}\n")

    return fitness


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
