from argparse import Namespace


def get_kwargs(args: Namespace) -> dict:
    kwargs = {}
    if isinstance(args, Namespace):
        for key, value in args.__dict__.items():
            if key not in ["name", "kwargs"] and value != 'delete':
                kwargs[key] = value
        if hasattr(args, "kwargs"):
            if isinstance(args.kwargs, dict):
                kwargs.update(args.kwargs)
            elif isinstance(args.kwargs, Namespace):
                kwargs.update(args.kwargs.__dict__)
            else:
                value = str(args.kwargs)
                raise ValueError("Expected dict or Namespace, but got " + value)
        return kwargs
    elif isinstance(args, dict):
        kwargs.update(args)
        if "name" in kwargs:
            del kwargs["name"]
        if "kwargs" in kwargs:
            del kwargs["kwargs"]
            kwargs.update(args["kwargs"])
