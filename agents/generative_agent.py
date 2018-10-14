from collections import OrderedDict
from typing import Dict, List, Tuple
from argparse import Namespace

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from models import Student
from models.data_generators import GenerativeProfessor
from agents.learning_agent import LearningAgent
from utils import get_optimizer
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

    def __init__(self,
                 students: List[Student],
                 args: Namespace) -> None:
        self.students = students
        self.professor = professor = GenerativeProfessor(args)
        self.professor_optimizer = get_optimizer(professor.parameters(),
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

        if args.c_d > 0:
            nclasses = 10  # TODO: change for other tasks
            self.classeye = torch.eye(nclasses)
            self.discriminator = discriminator = nn.Sequential(
                nn.Linear(nclasses * 2, 256),
                nn.ReLU(),
                nn.Linear(256, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            self.bce_loss = nn.BCELoss()
            self.discriminator_optimizer = optim.Adam(discriminator.parameters(),
                                                      lr=.001)

        self.debug = args.debug

    def to(self, device):
        self.professor.to(device)
        if self.c_d > 0:
            self.discriminator.to(device)
            self.bce_loss.to(device)
            self.classeye = self.classeye.to(device)

    def eval_student(self, student, **kwargs):
        return self.professor.eval_student(student, **kwargs)

    def discriminate(self, real_output, fake_output, target) -> torch.Tensor:
        cond = self.classeye[target]
        outputs = torch.cat((torch.cat((real_output.detach(), cond), dim=1),
                             torch.cat((fake_output, cond), dim=1)), dim=0)
        targets = torch.cat((torch.zeros_like(target),
                             torch.ones_like(target)), dim=0)
        return self.bce_loss(self.discriminator(outputs), targets.detach())

    def process(self, data, target) -> Tuple[List[float], Dict[str, float]]:
        c_nll, c_kl = self.c_nll, self.c_kl
        c_grad, c_cos, c_optim = self.c_grad, self.c_cos, self.c_optim
        c_hess = self.c_hess
        c_l2 = self.c_l2
        c_d = self.c_d

        student_loss = 0
        student_losses = []
        professor_loss = 0
        professor_losses = OrderedDict({})

        for coeff_name, loss_name in GenerativeAgent.LOSSES.items():
            if getattr(self, coeff_name) > 0:
                professor_losses[loss_name] = 0

        # --- Here it starts ---

        fake_data, _target = self.professor(target)
        real_outputs, fake_outputs = dict({}), dict({})

        if self.c_d > 0:
            professor_losses["Discriminator BCE - real"] = 0
            professor_losses["Discriminator BCE - fake"] = 0
            professor_losses["Discriminator BCE - gen "] = 0
            professor_losses["Discriminator avg - real"] = 0
            professor_losses["Discriminator avg - fake"] = 0
            professor_losses["Discriminator avg - gen "] = 0
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

                professor_losses["Discriminator BCE - real"] += loss_real.item()
                professor_losses["Discriminator avg - real"] += d_real_out.mean().item()

                professor_losses["Discriminator BCE - fake"] += loss_fake.item()
                professor_losses["Discriminator avg - fake"] += d_fake_out.mean().item()

            (discriminator_loss / len(self.students)).backward()
            self.discriminator_optimizer.step()

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
            s_loss = real_nll = real_nlls.mean()
            if c_l2 > 0:
                s_loss = s_loss + l2(student.parameters()) * c_l2

            student_loss += s_loss
            student_losses.append(s_loss.item())

            # Error in induced losses
            if c_nll > 0:
                # MSE between fake_nlls and real_nlls
                nll_loss = F.mse_loss(fake_nlls, real_nlls.detach()) * c_nll
                professor_loss += nll_loss
                professor_losses["NLL"] += nll_loss.item()

            # -- Error in KL between outputs --
            if c_kl > 0:
                # KL between fake_outputs and real_outputs
                target_p = F.softmax(real_output).detach()
                fake_logp = F.log_softmax(fake_output, dim=1)
                kldiv = F.kl_div(fake_logp, target_p) * c_kl
                professor_loss += kldiv
                professor_losses["KLdiv"] = kldiv.item()

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
                professor_losses["GradMSE"] += grad_loss.item()

            # -- Cosine distance between induced gradients --

            if c_cos > 0:
                cos_loss = 0
                for real_grads, fake_grads, _mask in grad_pairs:
                    cos_loss += cos(fake_grads, real_grads)
                cos_loss *= c_cos / ngrads
                professor_loss += cos_loss
                professor_losses["GradCos"] += cos_loss.item()

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
                professor_losses["NextNLL"] += optim_loss.item()

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
                professor_losses["HessVecMSE"] += hess_loss.item()

            if c_d > 0:

                d_gen_in = torch.cat((fake_output, label), dim=1)
                d_gen_out = self.discriminator(d_gen_in)
                d_gen_loss = self.bce_loss(d_gen_out, ones_for_d)
                professor_loss += d_gen_loss * c_d
                professor_losses["Discriminator BCE - gen "] += d_gen_loss.item()
                professor_losses["Discriminator avg - gen "] += d_gen_out.mean().item()

            # -- Backward now so we won't keep to much memory? Not for now

            # -- here ENDs computation fo current student

        # -- If there is some loss, improve teacher

        if torch.is_tensor(professor_loss):
            if torch.isnan(professor_loss).any().item():
                return None, None

            # Normalize this w.r.t. the number of students

            nstudents = float(len(self.students))
            professor_loss /= nstudents
            for key in professor_losses.keys():
                professor_losses[key] /= nstudents

            self.professor_optimizer.zero_grad()
            professor_loss.backward()

            if self.debug:
                pg_ratio = []
                for p in self.professor.parameters():
                    pg_ratio.append(f"{(p.data / p.grad.data).abs().mean().item():.2f}")
                print(pg_ratio)

            self.professor_optimizer.step()

        # -- Perform backpropagation through students

        for student in self.students:
            student.zero_grad()

        student_loss.backward()

        return student_losses, professor_losses
