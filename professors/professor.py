from typing import List, Tuple
from argparse import Namespace
import torch
from torch import Tensor
import torch.nn.functional as F

from utils import printer
from models import Student


class PostTrainProfessor:

    def eval_student(self, student, step, nsamples: int = None):
        raise NotImplementedError

    # ------------------------------------------------------------------
    #
    # Teaching modes.

    def teaching_modes(self) -> List[str]:
        """A professor might be able to train the students in different
        ways. This function should return all available modes."""
        return ["Teaching"]

    @property
    def teaching_mode(self) -> str:
        return self._teaching_mode

    @teaching_mode.setter
    def teaching_mode(self, value: str) -> None:
        if value not in self.teaching_modes():
            raise ValueError(f"Unknown training mode {value}.")
        self._teaching_mode = value


class Professor:

    def __init__(self, name: str = "PROFESSOR", verbose: int = 1):
        self.verbose = verbose
        self.info = printer(name, 1, verbose=verbose)
        self.debug = printer(name, 2, verbose=verbose)
        self._teaching_mode = "Student Acc."

    def process(self, data, target, data_idx=None) -> bool:
        raise NotImplementedError

    def end_epoch(self):
        pass

    def eval_student(self, student: Student,
                     step: int,
                     nsamples: int = None) -> Tuple[Tensor, float]:
        raise NotImplementedError

    def save_state(self, out_dir, epoch, data, target):
        raise NotImplementedError

    def post_train_professor(self, old_model=None) -> PostTrainProfessor:
        raise NotImplementedError


class DummyProfessor(Professor):

    def __init__(self, prof_args, device, _start_params=None):
        super(DummyProfessor, self).__init__("DUMMY", prof_args.verbose)
        self.nclasses = prof_args.nclasses
        self.in_size = prof_args.in_size
        self.device = device

    def process(self, data, target, data_idx=None) -> bool:
        return False

    def eval_student(self, student: Student,
                     step: int,
                     nsamples: int = None):
        nsamples = 16 if nsamples is None else nsamples
        fake_data = torch.randn(nsamples, *self.in_size, device=self.device)
        target = torch.randint(self.nclasses, (nsamples,), device=self.device,
                               dtype=torch.long)
        output = student(fake_data)
        loss = F.cross_entropy(output, target)
        with torch.no_grad():
            _, prediction = output.max(dim=1)
            acc = prediction.eq(target).sum().item() / nsamples
        return loss, acc

    def save_state(self, out_dir, epoch, data, target):
        pass

    def post_train_professor(self, old_model=None):
        if old_model is not None:
            return old_model
        return DummyProfessor(Namespace(nclasses=self.nclasses,
                                        in_size=self.in_size),
                              device=self.device)
