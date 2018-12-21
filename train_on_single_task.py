import os
import pickle
from copy import deepcopy
from argparse import Namespace
import concurrent.futures
from termcolor import colored as clr
import numpy as np

import torch
import torch.nn.functional as F

from utils import printer, args_to_dict
from utils import get_optimizer, print_conf
from loss_utils import l2
import professors
from models import get_model
from models import classifiers
from tasks.datasets import get_loaders

# -----------------------------------------------------------------------------
#
# Test a student on some data and return the accuracy, the average
# loss, and the confusion matrix. Data is given in a generator of
# (input, target) pairs.
#
# Warning: the model is left in `eval` mode.


def test(model, loader, device, verbose: int = 1):
    info = printer("TEST", 1, verbose=verbose)
    nclasses = model.nclasses  # Our classifiers provide this info
    conf_matrix = torch.zeros(nclasses, nclasses, device=device)
    loss, correct = .0, 0

    model.eval()
    with torch.no_grad():
        for data, target, data_idx in loader:
            data, target, data_idx = data.to(device), target.to(device), \
                                        data_idx.to(device)
            output = model(data)
            loss += F.cross_entropy(output, target, reduction="sum").item()
            _, prediction = output.max(1)
            correct += prediction.eq(target).sum().item()
            conf_matrix.put_(target * output.size(1) + prediction,
                             torch.ones(data.size(0), device=device),
                             accumulate=True)
    total = conf_matrix.sum().item()
    loss /= total  # The average loss on all data
    accuracy = (100. * correct / total)
    conf_matrix /= conf_matrix.sum(dim=1, keepdim=True) / 100.

    info("Cross entropy: ", clr(f"{loss:.3f}", "red"), "|",
         "Accuracy:", clr(f"{accuracy:.3f}%", "red"))
    print_conf(conf_matrix.cpu(), print_func=info)
    return accuracy, loss, conf_matrix


# -----------------------------------------------------------------------------
#
# Test a given professor by optimizing a few students on synthetic
# data starting from random parameters.

def test_professor(agent, loader, device, args, state_dict=None,
                   verbose: int = 1):
    all_accs = []
    nstudents = args.evaluation.nstudents
    nsteps = args.evaluation.teaching_steps
    info = printer("TEACH", 1, verbose=verbose)
    conf = None

    for sidx in range(1, nstudents + 1):
        info("Teaching a new one.", tags=[f"Student {sidx:d}/{nstudents:d}"])
        student = get_model(classifiers, args.student, in_size=args.in_size,
                            nclasses=args.nclasses).to(device)
        if state_dict is not None:
            student.load_state_dict(state_dict)
        student_optimizer = get_optimizer(student.parameters(),
                                          args.student_optimizer)
        crt_accs = []
        for step in range(nsteps):
            student_optimizer.zero_grad()
            synthetic_loss, synthetic_acc = agent.eval_student(student, step)
            if args.c_l2 > 0:
                l2_loss = l2(student.parameters()) * args.c_l2
                full_loss = synthetic_loss + l2_loss
            else:
                l2_loss = None
                full_loss = synthetic_loss
            full_loss.backward()
            student_optimizer.step()
            if (step + 1) % args.evaluation.eval_freq == 0:
                acc, loss, conf = test(student, loader, device, verbose=False)
                student.train()
                crt_accs.append(acc)
                info("Syn. NLL:", f"{synthetic_loss.item():.3f};",
                     "Syn. Acc:", f"{synthetic_acc:.2f}%;",
                     "" if l2_loss is None else f"L2: {l2_loss.item():.3f};",
                     clr(f"Real NLL: {loss:.3f}", "yellow") + ";",
                     clr(f"Real Acc: {acc:.1f}%", "yellow"),
                     tags=[f"Student {sidx:d}/{nstudents:d}",
                           f"{step + 1: 4d}/{nsteps:d}"])
        print_conf(conf, print_func=info)
        if len(all_accs) > 25:
            crt_accs = crt_accs[-25:]  # Keep the last scores only
        all_accs.append(np.mean(crt_accs))
    final_score = np.mean(all_accs)
    info(clr(f" >>> Final score = {final_score:.3f} <<<", "yellow"))
    return final_score


# -----------------------------------------------------------------------------
#
# The main script to be run on any agent.

def run(args: Namespace):

    if args.wandb:
        import wandb
        wandb.init()
        wandb.config.update(args_to_dict(args))

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
                    args.test_batch_size, in_size=tuple(args.in_size),
                    limit=args.dataset_limit,
                    normalize=False)
    train_loader.to(device)
    test_loader.to(device)

    some_batch, some_batch_idx = train_loader.sample(nsamples=32)

    in_size, nclasses, nrmlz = data_info
    args.in_size = in_size
    args.nclasses = nclasses

    info("Prepared loaders for", clr(args.dataset, "red"),
         "with", nclasses, "classes.")

    # -------------------------------------------------------------------------
    #
    # If evaluation is performed on a fixed set of parameters, save them.

    if not args.random_params:
        dummy_student = get_model(classifiers, args.student,
                                  in_size=in_size,
                                  nclasses=nclasses).to(device)
        start_params = deepcopy(dummy_student.state_dict())
        del dummy_student
        info("Saved a fixed set of parameters for the student model.")
    else:
        start_params = None

    # -------------------------------------------------------------------------
    #
    # Initialize professor.

    prof_args = args.professor

    if not hasattr(prof_args, "verbose"):
        prof_args.verbose = verbose

    prof_args.c_l2 = args.c_l2
    prof_args.in_size = in_size
    prof_args.nclasses = nclasses
    prof_args.nrmlz = nrmlz
    prof_args.wandb = args.wandb
    prof_args.student = args.student
    prof_args.student_optimizer = args.student_optimizer

    # TODO: Other professors
    Professor = getattr(professors, prof_args.name)
    if args.professor.generator.name == 'MemGenerator':
        professor = Professor(prof_args, device, start_params=start_params,
                                ds_size = train_loader.ds_size)
    else:
        professor = Professor(prof_args, device, start_params=start_params)

    # -------------------------------------------------------------------------
    #
    # Start training

    seen_examples = 0
    last_professor_eval = 0
    best_fitness, best_idx, scores, scores_at = None, None, [], []
    found_nan, should_stop = False, False

    eval_professor = None
    async_result = None

    if args.evaluation.continuous and not args.evaluation.async:
        raise ValueError("Continuous eval is available in async mode only.")
    if args.evaluation.async:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    for epoch in range(1, args.nepochs + 1):
        for _batch_idx, (data, target, data_idx) in enumerate(train_loader):
            data, target, data_idx = data.to(device), target.to(device),\
                                        data_idx.to(device)
            found_nan = professor.process(data, target, data_idx)
            seen_examples += len(data)

            if found_nan is None:
                info("Found NaN. Exiting.")
                found_nan = should_stop = True
                break

            new_score = False
            eval_professor = professor.post_train_professor()
            if async_result is not None:
                seen_at, result = async_result
                if result.done():
                    score = result.result()
                    scores.append(score)
                    scores_at.append(seen_at)
                    if args.wandb:
                        wandb.log({"Step": seen_at, "score": score})
                    async_result = None
                    new_score = True
                    info("Ended evaluation at", seen_at, "steps with",
                         clr(f"{score:.3f}%", "white", "on_magenta"),
                         tags=["TEACH"])

            if async_result is not None:
                pass
            elif args.evaluation.continuous:
                info("Started evaluation at", seen_examples, "steps",
                     tags=["TEACH"])
                eval_professor = professor.post_train_professor(
                    old_model=eval_professor)
                result = executor.submit(test_professor,
                                         eval_professor, test_loader, device, args,
                                         state_dict=start_params,
                                         verbose=0)
                async_result = (seen_examples, result)
            elif seen_examples - last_professor_eval >= args.evaluation.freq:
                info("Started evaluation at", seen_examples, "steps",
                     tags=["TEACH"])
                if args.evaluation.async:
                    eval_professor = professor.post_train_professor()
                    result = executor.submit(test_professor,
                                             eval_professor, test_loader,
                                             device, args,
                                             state_dict=start_params,
                                             verbose=0)
                    async_result = (seen_examples, result)
                else:
                    score = test_professor(eval_professor, test_loader, device,
                                           args, state_dict=start_params)
                    scores.append(score)
                    scores_at.append(seen_examples)
                    if args.wandb:
                        wandb.log({"Step": seen_at, "score": score})
                    info("Evaluation at", seen_examples, "steps ended with",
                         clr(f"{score:.3f}%", "white", "on_magenta"),
                         tags=["TEACH"])
                last_professor_eval += args.evaluation.freq

            if new_score and len(scores) >= 10:
                new_avg = np.mean(scores[-10:])
                if best_fitness is None or new_avg > best_fitness:
                    best_fitness, best_idx = new_avg, len(scores)

                if best_idx + 100 <= len(scores):
                    info("Early stopping: 100 evaluations, no improvement")
                    should_stop = True
                    break

        if should_stop:
            break
        professor.end_epoch()

        professor.save_state(args.out_dir, epoch, 
                                data_idx=some_batch_idx, **some_batch)
        with open(os.path.join(args.out_dir, f"eval.th"), 'wb') as handle:
            pickle.dump({"scores": scores, "seen": scores_at},
                        handle, protocol=pickle.HIGHEST_PROTOCOL)

        info(f"Ended epoch {epoch:2d} / {args.nepochs:d}.")

    if args.evaluation.async:
        executor.shutdown()  # It waits for any async evaluation to end
        if async_result is not None:
            seen_at, result = async_result
            score = result.result()
            scores.append(score)
            scores_at.append(seen_at)
            if args.wandb:
                wandb.log({"step": seen_at, "score": score})
            if len(scores) >= 10:
                new_avg = np.mean(scores[-10:])
                if best_fitness is None or new_avg > best_fitness:
                    best_fitness = new_avg
            info("Evaluation at", seen_at, "steps",
                 clr(f"{score:.3f}%", "white", "on_magenta"),
                 tags=["TEACH"])
            with open(os.path.join(args.out_dir, f"eval.th"), 'wb') as handle:
                pickle.dump({"scores": scores, "seen": scores_at},
                            handle, protocol=pickle.HIGHEST_PROTOCOL)

    if found_nan:
        fitness = -1.0
    elif best_fitness is None:
        fitness = np.mean(scores)
    else:
        fitness = best_fitness

    summary = {
        "best": fitness,
        "last-5": np.mean(scores[-5:]),
    }

    with open(os.path.join(args.out_dir, 'summary.pkl'), 'wb') as handler:
        pickle.dump(summary, handler, pickle.HIGHEST_PROTOCOL)

    info(f"Final fitness: {fitness:.3f}")

    with open(os.path.join(args.out_dir, "fitness"), "w") as handler:
        handler.write(f"{fitness:f}\n")

    if args.wandb:
        wandb.run.summary["fitness"] = fitness

    return fitness


# -- The usual main function to be used with liftoff

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
