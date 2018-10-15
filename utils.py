from typing import Iterator, List, Tuple, Union
from argparse import Namespace
from termcolor import colored as clr
import torch.optim as optim
import torch.nn as nn


def print_nparams(model: nn.Module, name: str = "Model") -> None:
    nparams = sum(p.nelement() for p in model.parameters())
    print(f"[MODEL] {name:s} has {clr(f'{nparams:d}', 'yellow'):s} params.")


def get_optimizer(parameters, opt_args: Namespace) -> optim.Optimizer:
    cfgargs = get_kwargs(opt_args)
    print("[MAIN_] The arguments for the", clr(opt_args.name, 'red'),
          "optimizer: ", cfgargs)
    return getattr(optim, opt_args.name)(parameters, **cfgargs)


def get_kwargs(args: Union[Namespace, dict],
               ignore: List[str] = None) -> dict:
    if ignore is None:
        ignore = ["name", "kwargs"]
    else:
        ignore.append("kwargs")

    kwargs = {}
    if isinstance(args, Namespace):
        for key, value in args.__dict__.items():
            if key not in ignore and value != 'delete':
                kwargs[key] = value
        if hasattr(args, "kwargs"):
            if isinstance(args.kwargs, dict):
                kwargs.update(args.kwargs)
            elif isinstance(args.kwargs, Namespace):
                kwargs.update(args.kwargs.__dict__)
            else:
                msg = "Expected dict or Namespace, but got " + str(args.kwargs)
                raise ValueError(msg)
    elif isinstance(args, dict):
        kwargs.update(args)
        for key in ignore:
            if key in kwargs:
                del kwargs[key]
        if "kwargs" in args:
            kwargs.update(args["kwargs"])
    else:
        raise ValueError("Expected Namespace or dict")

    return kwargs
