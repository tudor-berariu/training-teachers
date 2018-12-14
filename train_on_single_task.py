from argparse import Namespace
import concurrent.futures
from copy import deepcopy
import os
import pickle
from typing import List
import numpy as np
from termcolor import colored as clr
import torch
import torch.nn.functional as F

from utils import printer
from utils import get_optimizer, print_conf
from utils import best_and_last
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
            data, target = data.to(device), target.to(device)
            data_idx = data_idx.to(device)
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

def test_professor(professor, loader, device, args, state_dict=None,
                   verbose: int = 1):
    scores = {}  # type: Dict[str, float]
    for teaching_mode in professor.teaching_modes():
        professor.teaching_mode = teaching_mode
        scores[teaching_mode] = _test_professor(professor,
                                                loader, device, args,
                                                state_dict=state_dict,
                                                verbose=verbose)
    return scores


def _test_professor(professor, loader, device, args, state_dict=None,
                    verbose: int = 1):
    all_accs = []  # type: List[float]
    nstudents = args.evaluation.nstudents
    nsteps = args.evaluation.teaching_steps
    lastk = args.evaluation.last_k
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
            syn_loss, syn_acc = professor.eval_student(student, step)
            if args.c_l2 > 0:
                l2_loss = l2(student.parameters()) * args.c_l2
                full_loss = syn_loss + l2_loss
            else:
                l2_loss = None
                full_loss = syn_loss
            full_loss.backward()
            student_optimizer.step()
            if (step + 1) % args.evaluation.eval_freq == 0:
                acc, loss, conf = test(student, loader, device, verbose=False)
                student.train()
                crt_accs.append(acc)
                info("Syn. Loss:", f"{syn_loss.item():.5f};",
                     "Syn. Acc:", f"{syn_acc:.2f}%;",
                     "" if l2_loss is None else f"L2: {l2_loss.item():.3f};",
                     clr(f"Real NLL: {loss:.4f}", "yellow") + ";",
                     clr(f"Real Acc: {acc:.1f}%", "yellow"),
                     tags=[f"Student {sidx:d}/{nstudents:d}",
                           f"{step + 1: 4d}/{nsteps:d}"])
        print_conf(conf, print_func=info)
        if len(all_accs) > lastk:
            crt_accs = crt_accs[-lastk:]  # Keep the last scores only
        all_accs.append(np.mean(crt_accs))
    final_score = np.mean(all_accs)
    info(clr(f" >>> Final score = {final_score:.3f} <<<", "yellow"))
    return final_score


def log_new_scores(seen_at, new_scores, scores, scores_at,
                   print_func, writer=None) -> None:
    msgs = []
    for teaching_mode, score in new_scores.items():
        scores.setdefault(teaching_mode, []).append(score)
        if writer is not None:
            writer.add_scalar(f'teach/{teaching_mode:s}', score, seen_at)
        msgs.append(f"{teaching_mode:s} = " +
                    clr(f"{score:.3f}%", "white", "on_magenta"))
    scores_at.append(seen_at)

    print_func("Ended evaluation at", seen_at, "steps with",
               "; ".join(msgs), tags=["TEACH"])

# -----------------------------------------------------------------------------
#
# The main script to be run on any agent.


def run(args: Namespace):

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        # TODO: is there a better way to get rid of old files?
        to_delete = [f for f in os.listdir() if "tfevents" in f]
        for tfevent_file in to_delete:
            os.remove(tfevent_file)
        writer = SummaryWriter(log_dir=args.out_dir)
    else:
        writer = None

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

    some_batch = train_loader.sample(nsamples=32)

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
    prof_args.writer = writer
    prof_args.student = args.student
    prof_args.student_optimizer = args.student_optimizer

    Professor = getattr(professors, prof_args.name)
    if args.professor.generator.name == 'MemGenerator':
        professor = Professor(prof_args, device, start_params=start_params,
                              ds_size=train_loader.ds_size)
    else:
        professor = Professor(prof_args, device, start_params=start_params)

    # -------------------------------------------------------------------------
    #
    # Start training

    seen_examples = 0
    last_professor_eval = 0
    scores, scores_at = {}, []
    found_nan, should_stop = False, False

    eval_professor = None
    async_result = None  # type: Optional[tuple]

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

            if found_nan:
                info("Found NaN. Exiting.")
                found_nan = should_stop = True
                break

            # -----------------------------------------------------------------
            # --- All above is just evaluation handling

            if async_result is not None:
                # -- The asynchronous evaluation ended.
                seen_at, result = async_result
                if result.done():
                    new_scores = result.result()  # type: Dict[str, float]
                    log_new_scores(seen_at, new_scores, scores, scores_at,
                                   info, writer)
                    async_result = None

            if async_result is not None:
                # -- It means there's an async evaluation currently running.
                pass
            elif args.evaluation.continuous:
                # -- It means we should start evaluating the model.
                info("Started evaluation at", seen_examples, "steps",
                     tags=["TEACH"])
                eval_professor = professor.post_train_professor(
                    old_model=eval_professor)
                result = executor.submit(test_professor,
                                         eval_professor, test_loader, device,
                                         args,
                                         state_dict=start_params,
                                         verbose=0)
                async_result = (seen_examples, result)
            elif seen_examples - last_professor_eval >= args.evaluation.freq:
                # -- It means we should start evaluating the model.
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
                    eval_professor = professor.post_train_professor(
                        old_model=eval_professor)
                    new_scores = test_professor(eval_professor, test_loader,
                                                device,
                                                args, state_dict=start_params)

                    log_new_scores(seen_examples, new_scores, scores, scores_at,
                                   info, writer)
                last_professor_eval += args.evaluation.freq

            # -----------------------------------------------------------------

        if should_stop:
            break

        professor.end_epoch()
        professor.save_state(args.out_dir, epoch, **some_batch)
        with open(os.path.join(args.out_dir, f"eval.th"), 'wb') as handle:
            pickle.dump({"scores": scores, "seen": scores_at},
                        handle, protocol=pickle.HIGHEST_PROTOCOL)

        info(f"Ended epoch {epoch:2d} / {args.nepochs:d}.")

    if args.evaluation.async:
        executor.shutdown()  # It waits for any async evaluation to end
        if async_result is not None:
            seen_at, result = async_result
            new_scores = result.result()
            log_new_scores(seen_at, new_scores, scores, scores_at,
                           info, writer)
            with open(os.path.join(args.out_dir, f"eval.th"), 'wb') as handle:
                pickle.dump({"scores": scores, "seen": scores_at},
                            handle, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {}
    for attr, values in scores.items():
        best, last = best_and_last(values, 5)
        summary[f"{attr:s}:best"] = best
        summary[f"{attr:s}:last"] = last

    with open(os.path.join(args.out_dir, 'summary.pkl'), 'wb') as handler:
        pickle.dump(summary, handler, pickle.HIGHEST_PROTOCOL)

    if args.tensorboard:
        writer.export_scalars_to_json(os.path.join(args.out_dir, "trace.json"))
        writer.close()

    info(f"Final fitness: {best:.3f}")


# -- The usual main function to be used with liftoff

def main() -> None:
    # Reading args
    from liftoff.config import read_config
    args = read_config()  # type: Namespace

    if not hasattr(args, "out_dir"):
        raise Exception(f"Out directory was not provided.")
    elif not os.path.isdir(args.out_dir):
        raise Exception(f"Directory {args.out_dir} does not exist.")

    if not hasattr(args, "run_id"):
        args.run_id = 0

    run(args)


if __name__ == "__main__":
    main()
