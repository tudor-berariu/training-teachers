import torch
import torch.nn.functional as F

from utils import printer


class Professor:

    def __init__(self, name: str = "PROFESSOR", verbose: int = 1):
        self.verbose = verbose
        self.info = printer(name, 1, verbose=verbose)
        self.debug = printer(name, 2, verbose=verbose)

    def process(self, data, target) -> bool:
        raise NotImplementedError

    def eval_student(self, student, step):
        raise NotImplementedError

    def save_state(self, out_dir, epoch, data, target):
        raise NotImplementedError


class DummyProfessor(Professor):

    def __init__(self, prof_args, device, start_params=None):
        super(DummyProfessor, self).__init__("DUMMY", prof_args.verbose)
        self.nclasses = prof_args.nclasses
        self.in_size = prof_args.in_size
        self.device = device

    def process(self, data, target) -> bool:
        return False

    def eval_student(self, student, step):
        fake_data = torch.randn(16, *self.in_size, device=self.device)
        target = torch.randint(self.nclasses, (16,), device=self.device,
                               dtype=torch.long)
        output = student(fake_data)
        loss = F.cross_entropy(output, target)
        with torch.no_grad():
            _, prediction = output.max(dim=1)
            acc = prediction.eq(target).sum().item() / 16
        return loss, acc

    def save_state(self, out_dir, epoch, data, target):
        pass
