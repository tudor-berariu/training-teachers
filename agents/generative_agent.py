from collections import OrderedDict
from typing import Dict, List, Tuple
from argparse import Namespace
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from models import Student
from models.data_generators import SyntheticDataGenerator
from models.data_generators import SyntheticDataDiscriminator
from agents.learning_agent import LearningAgent

from utils import get_optimizer, print_nparams, get_kwargs
from loss_utils import cos, mse, l2


def grad_of(outputs, inputs, grad_outputs=None):
    return autograd.grad(outputs, inputs,
                         grad_outputs=grad_outputs,
                         create_graph=True,
                         retain_graph=True,
                         only_inputs=True)


class GenerativeAgent(LearningAgent):

    LOSSES = OrderedDict({"c_nll": "NLL",
                          "c_kl": "KLdiv",
                          "c_grad": "GradMSE",
                          "c_cos": "GradCos",
                          "c_optim": "NextNLL",
                          "c_hess": "HessVecMSE",
                          "c_d": "Discriminator"})

    AG_KWARGS = {"create_graph": True,
                 "retain_graph": True,
                 "only_inputs": True}

    def __init__(self, students: List[Student], args: Namespace) -> None:
        self.students = students
        self.args = args
        self.in_size = in_size = args.in_size
        self.nclasses = nclasses = args.nclasses

        DataGenerator = eval(args.data_generator.name)
        self.generator = DataGenerator(in_size=in_size, nclasses=nclasses,
                                       **get_kwargs(args.data_generator))
        print_nparams(self.generator, name="Generator:0")

        self.generator_optimizer = get_optimizer(self.generator.parameters(),
                                                 args.optimizer)

        self.c_nll = args.c_nll  # Coefficient for MSELoss between NLLs
        self.c_kl = args.c_kl  # Coefficient for KLDiv between outputs
        self.c_grad = args.c_grad  # Coefficient for MSELoss between gradients
        self.c_cos = args.c_cos  # Coefficient for the cos between gradients
        self.batch_grad = args.batch_grad  # Compute gradient w.r.t. full batch
        self.c_optim = args.c_optim  # Coefficient for the nll of moved params
        self.c_hess = args.c_hess  # Coefficient for the MSE between Hv prods
        self.c_d = args.c_d  # Coefficient for the discriminator
        self.c_l2 = args.c_l2

        self.full_grad = args.full_grad

        if args.c_d > 0:
            self.classeye = torch.eye(nclasses)
            self.discriminator = disc = SyntheticDataDiscriminator(nclasses)
            self.bce_loss = nn.BCELoss()
            self.discriminator_optimizer = optim.Adam(disc.parameters(), lr=.001)

        self.old_generator = None
        self.crt_device = None
        self.eval_samples = args.eval_samples
        self.debug = args.debug

    def to(self, device):
        self.crt_device = device
        self.generator.to(device)
        if self.c_d > 0:
            self.discriminator.to(device)
            self.bce_loss.to(device)
            self.classeye = self.classeye.to(device)
        if self.old_generator is not None:
            self.old_generator.to(device)

    def eval_student(self, student: Student, nsamples: int = None) -> torch.Tensor:
        if nsamples is None:
            nsamples = self.eval_samples
        data, target = self.generator(nsamples=nsamples)
        output = student(data)
        return F.cross_entropy(output, target)

    def end_task(self, is_last: bool = False) -> None:
        if is_last:
            return

        args = self.args
        self.old_generator = self.generator
        self.old_generator.eval()
        del self.generator_optimizer

        data_generator_cfg = args.data_generator
        DataGenerator = eval(data_generator_cfg.name)
        self.generator = DataGenerator(**get_kwargs(data_generator_cfg))
        print_nparams(self.generator, name="Generator")

        if self.crt_device:
            self.generator.to(self.crt_device)
        self.generator_optimizer = get_optimizer(self.generator.parameters(),
                                                 args.optimizer)
        if args.c_d > 0:
            del self.discriminator
            del self.discriminator_optimizer
            nclasses = self.nclasses
            self.classeye = torch.eye(nclasses)
            self.discriminator = disc = SyntheticDataDiscriminator(nclasses)
            self.discriminator_optimizer = optim.Adam(disc.parameters(), lr=.001)

    def process(self, data, target) -> Tuple[List[float], Dict[str, float]]:
        c_nll, c_kl = self.c_nll, self.c_kl
        c_grad, c_cos, c_optim = self.c_grad, self.c_cos, self.c_optim
        c_hess = self.c_hess
        c_l2 = self.c_l2
        c_d = self.c_d

        nstudents = len(self.students)

        student_losses = []
        generator_losses = OrderedDict({})

        for coeff_name, loss_name in GenerativeAgent.LOSSES.items():
            if getattr(self, coeff_name) > 0:
                generator_losses[loss_name] = 0

        # --- Here it starts ---

        # -- Distil old generator in current one?

        if self.old_generator is not None:
            with torch.no_grad():
                prev_data, prev_target = self.old_generator(nsamples=len(data))
            data = torch.cat((data, prev_data.detach()), dim=0)
            target = torch.cat((target, prev_target.detach()), dim=0)

        # -- Generate data and apply costs

        fake_data, _target = self.generator(target)
        real_outputs, fake_outputs = dict({}), dict({})

        if self.c_d > 0:
            generator_losses["Discriminator BCE - real"] = 0
            generator_losses["Discriminator BCE - fake"] = 0
            generator_losses["Discriminator BCE - gen "] = 0
            generator_losses["Discriminator avg - real"] = 0
            generator_losses["Discriminator avg - fake"] = 0
            generator_losses["Discriminator avg - gen "] = 0
            label = self.classeye[target]
            discriminator_loss = 0
            self.discriminator_optimizer.zero_grad()

            ones_for_d = torch.ones(target.size(), device=target.device).unsqueeze(1)
            zeros_for_d = torch.zeros(target.size(), device=target.device).unsqueeze(1)

            for idx, student in enumerate(self.students):
                real_outputs[idx] = real_output = student(data)
                fake_outputs[idx] = fake_output = student(fake_data)

                d_real_in = torch.cat((real_output.detach(), label), dim=1)
                d_real_out = self.discriminator(d_real_in)
                loss_real = self.bce_loss(d_real_out, ones_for_d)

                d_fake_in = torch.cat((fake_output.detach(), label), dim=1)
                d_fake_out = self.discriminator(d_fake_in)
                loss_fake = self.bce_loss(d_fake_out, zeros_for_d)

                discriminator_loss += loss_real + loss_fake

                generator_losses["Discriminator BCE - real"] += loss_real.item()
                generator_losses["Discriminator avg - real"] += d_real_out.mean().item()

                generator_losses["Discriminator BCE - fake"] += loss_fake.item()
                generator_losses["Discriminator avg - fake"] += d_fake_out.mean().item()

            (discriminator_loss / len(self.students)).backward()
            self.discriminator_optimizer.step()

        self.generator.zero_grad()
        optimize_generator = False
        for s_idx, student in enumerate(self.students):
            if self.c_d > 0:
                real_output = real_outputs[s_idx]
                fake_output = fake_outputs[s_idx]
            else:
                real_output = student(data)
                fake_output = student(fake_data)
            real_nlls = F.cross_entropy(real_output, target, reduction="none")
            fake_nlls = F.cross_entropy(fake_output, target, reduction="none")

            # Student's negative log likelihood on the real data
            student_loss = real_nll = real_nlls.mean()
            if c_l2 > 0:
                student_loss = student_loss + l2(student.parameters()) * c_l2
            student_losses.append(student_loss.item())

            professor_loss = 0

            # Error in induced losses
            if c_nll > 0:
                # MSE between fake_nlls and real_nlls
                nll_loss = F.mse_loss(fake_nlls, real_nlls.detach()) * c_nll
                professor_loss += nll_loss
                generator_losses["NLL"] += nll_loss.item()

            # -- Error in KL between outputs --
            if c_kl > 0:
                # KL between fake_outputs and real_outputs
                target_p = F.softmax(real_output, dim=1).detach()
                fake_logp = F.log_softmax(fake_output, dim=1)
                kldiv = F.kl_div(fake_logp, target_p) * c_kl
                professor_loss += kldiv
                generator_losses["KLdiv"] = kldiv.item()

            if c_grad > 0 or c_cos > 0 or c_hess > 0 or c_optim > 0:
                grad_pairs = []
                if self.batch_grad:
                    real_g = grad_of(real_nll, student.parameters())
                    fake_g = grad_of(fake_nlls.mean(), student.parameters())
                    grad_pairs.append((real_g, fake_g, None))
                else:
                    for class_idx in range(real_output.size(1)):
                        mask = (target == class_idx)
                        if mask.any().item():
                            real_nll_i = real_nlls[mask].mean()
                            fake_nll_i = fake_nlls[mask].mean()
                            real_g = grad_of(real_nll_i, student.parameters())
                            fake_g = grad_of(fake_nll_i, student.parameters())
                            grad_pairs.append((real_g, fake_g, mask))

                ngrads = len(grad_pairs)

            # -- MSError between induced gradients --

            if c_grad > 0:
                grad_loss = 0
                for real_g, fake_g, _mask in grad_pairs:
                    grad_loss += mse(fake_g, real_g)
                grad_loss *= c_grad / ngrads
                professor_loss += grad_loss
                generator_losses["GradMSE"] += grad_loss.item()

            # -- Cosine distance between induced gradients --

            if c_cos > 0:
                cos_loss = 0
                for real_grads, fake_grads, _mask in grad_pairs:
                    cos_loss += cos(fake_grads, real_grads)
                cos_loss *= c_cos / ngrads
                professor_loss += cos_loss
                generator_losses["GradCos"] += cos_loss.item()

            # -- Cross-entropy of displaced parameters --

            if c_optim > 0:
                optim_loss = 0
                for _real_grads, fake_grads, mask in grad_pairs:
                    new_params = OrderedDict({})
                    pg_pairs = zip(student.named_parameters(), fake_grads)
                    for (name, param), grad in pg_pairs:
                        new_params[name] = param.detach() + grad
                    if mask is None:
                        new_output = student(data, params=new_params)
                        optim_loss += F.cross_entropy(new_output, target)
                    else:
                        data_i, target_i = data[mask], target[mask]
                        new_output = student(data_i, params=new_params)
                        optim_loss += F.cross_entropy(new_output, target_i)

                optim_loss *= c_optim / ngrads
                professor_loss += optim_loss
                generator_losses["NextNLL"] += optim_loss.item()

            # -- MSError between Hessian-vector products --

            if c_hess > 0:
                hess_loss = 0
                for real_grads, fake_grads, _mask in grad_pairs:
                    rand_v = [torch.bernoulli(torch.rand_like(g))
                              for g in real_grads]
                    target_Hv = grad_of(real_grads, student.parameters(),
                                        grad_outputs=rand_v)
                    fake_Hv = grad_of(fake_grads, student.parameters(),
                                      grad_outputs=rand_v)
                    hess_loss += mse(fake_Hv, target_Hv)

                hess_loss *= c_hess / ngrads
                professor_loss += hess_loss
                generator_losses["HessVecMSE"] += hess_loss.item()

            if c_d > 0:

                d_gen_in = torch.cat((fake_output, label), dim=1)
                d_gen_out = self.discriminator(d_gen_in)
                d_gen_loss = self.bce_loss(d_gen_out, ones_for_d)
                professor_loss += d_gen_loss * c_d
                generator_losses["Discriminator BCE - gen "] += d_gen_loss.item()
                generator_losses["Discriminator avg - gen "] += d_gen_out.mean().item()

            # -- Backward now so we won't keep to much memory? Not for now

            if torch.is_tensor(professor_loss):
                if torch.is_tensor(professor_loss):
                    if torch.isnan(professor_loss).any().item():
                        return None, None
                optimize_generator = True
                professor_loss /= nstudents
                professor_loss.backward(retain_graph=True)

            # -- Perform backpropagation through current student --

            if not self.full_grad:
                student.zero_grad()
            student_loss.backward()

            # -- here ENDs computation fo current student

        # -- If there is some loss, improve teacher

        if optimize_generator:
            if self.debug:
                tuples = []
                with torch.no_grad():
                    for name, param in self.generator.named_parameters():
                        p_mean = param.data.abs().mean().item()
                        g_mean = param.grad.data.abs().mean().item()
                        pg_ratio = param.data.abs() / (param.grad.data.abs() + 1e-9)
                        pg_ratio = pg_ratio.mean().item()
                        tuples.append((name, p_mean, g_mean, pg_ratio))
                print(tabulate(tuples))

            self.generator_optimizer.step()

        for key in generator_losses.keys():
            generator_losses[key] /= nstudents

        return student_losses, generator_losses
