from itertools import chain
from collections import OrderedDict
from typing import List
import os.path
from argparse import Namespace
from tabulate import tabulate

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from models import Student
from models import get_model
from models import generative
from agents.learning_agent import LearningAgent

from utils import get_optimizer, print_nparams, grad_info
from loss_utils import cos, mse, l2

from torchvision.utils import save_image


def grad_of(outputs, inputs, grad_outputs=None):
    """Call autograd.grad with create & retain graph, and ignore other
       leaf variables.

    """
    return autograd.grad(outputs, inputs,
                         grad_outputs=grad_outputs,
                         create_graph=True,
                         retain_graph=True,
                         only_inputs=True)


class GenerativeAgent(LearningAgent):
    #pylint: disable=too-many-instance-attributes

    def __init__(self, students: List[Student], args: Namespace) -> None:
        self.students = students
        self.args = args
        self.in_size = args.in_size
        self.nclasses = args.nclasses
        self.generator_idx = 0

        self.generator, self.encoder, self.discriminator = None, None, None
        self.d_optimizer, self.prof_optimizer = None, None
        self.old_generator = None

        self.classeye = torch.eye(args.nclasses)
        self._create_components()

        self.coeffs = coeffs = Namespace()

        self.coeffs.c_nll = args.c_nll
        self.coeffs.c_kl = args.c_kl
        self.coeffs.c_grad_mse = args.c_grad_mse
        self.coeffs.c_grad_cos = args.c_grad_cos
        self.coeffs.c_next_nll = args.c_next_nll
        self.coeffs.c_next_kl = args.c_next_kl
        self.coeffs.c_hess = args.c_hess
        self.coeffs.c_d = args.c_d
        self.coeffs.c_recon = args.c_recon
        self.coeffs.c_l2 = args.c_l2
        self.coeffs.c_latent_kl = args.c_latent_kl
        self.coeffs.next_lr = args.next_lr

        w_grad = ["grad_mse", "grad_cos", "next_kl", "next_nll", "hess"]
        self.need_grad = any(getattr(coeffs, "c_" + n) > 0 for n in w_grad)
        self.grad_type = args.grad_type
        assert self.grad_type in ["batch", "example", "class"]

        self.old_generator = None
        self.crt_device = None
        self.eval_samples = args.eval_samples
        self.debug = args.debug

    def _create_components(self):
        args = self.args
        self.generator = get_model(generative, args.generator,
                                   in_size=self.in_size,
                                   nclasses=self.nclasses,
                                   nz=args.nz)
        print_nparams(self.generator, name=f"Generator:{self.generator_idx:d}")

        if hasattr(args, "encoder") and args.c_latent_kl > 0:
            self.encoder = get_model(generative, args.encoder,
                                     in_size=args.in_size, nz=args.nz)
            all_params = chain(self.encoder.parameters(),
                               self.generator.parameters())
            print_nparams(self.encoder, name=f"Encoder")
        else:
            all_params = self.generator.parameters()

        self.prof_optimizer = get_optimizer(all_params, args.optimizer)

        if hasattr(args, "discriminator") and args.c_d > 0:
            discriminator = get_model(generative, args.discriminator,
                                      nclasses=args.nclasses)
            self.discriminator = discriminator
            self.bce_loss = nn.BCELoss()
            self.d_optimizer = optim.Adam(discriminator.parameters(), lr=.001)

    def to(self, device):  # pylint: disable=invalid-name
        self.crt_device = device
        self.generator.to(device)
        if self.discriminator is not None:
            self.discriminator.to(device)
            self.bce_loss.to(device)
            self.classeye = self.classeye.to(device)
        if self.encoder is not None:
            self.encoder.to(device)
        if self.old_generator is not None:
            self.old_generator.to(device)

    def eval_student(self, student: Student, nsamples: int = None) -> Tensor:
        if nsamples is None:
            nsamples = self.eval_samples
        data, target = self.generator(nsamples=nsamples)
        output = student(data)

        pred = output.max(1, keepdim=True)[1]
        correct = pred.eq(target.view_as(pred)).sum().item()

        return F.cross_entropy(output, target), (correct / len(data) * 100)

    def save_state(self, out_path: str, epoch_no: int,
                   data=None, target=None) -> None:
        generator = self.generator
        encoder = self.encoder

        # 1. save parameters
        torch.save((generator.state_dict(),
                    None if encoder is None else encoder.state_dict()),
                   os.path.join(out_path, f"params_{epoch_no:04d}.th"))

        # 2. generate_some_images
        with torch.no_grad():
            fake_data, _ = self.generator(nsamples=64)
            save_image((fake_data.cpu() + 1) / 2,
                       os.path.join(out_path, f"samples_{epoch_no:04d}.png"))

        # 3. save some comparisons

        with torch.no_grad():
            mean, log_var = encoder(data)
            fake_data, _target = generator(target, mean=mean, log_var=log_var)
            all_data = torch.cat((data, fake_data), dim=0).cpu()
            save_image(all_data,
                       os.path.join(out_path, f"recons_{epoch_no:04d}.png"))

    def end_task(self, is_last: bool = False) -> None:
        if is_last:
            return
        self.old_generator = self.generator
        self.old_generator.eval()

        self.generator, self.encoder, self.discriminator = None, None, None
        self.prof_optimizer, self.d_optimizer = None, None
        self.generator_idx += 1

        self._create_components()

    # pylint: disable=too-many-statements,too-many-branches,too-many-locals
    def process(self, data, target, nrmlz):
        nstudents = len(self.students)
        student_losses = []
        info = OrderedDict({})
        coeffs = self.coeffs
        encoder = self.encoder
        generator = self.generator

        # -----------------------------------------------------------------
        #
        # If there is an old generator, we just add some synthetic
        # data for previous tasks. The batch size gets doubled.

        if self.old_generator is not None:
            with torch.no_grad():
                prev_data, prev_target = self.old_generator(nsamples=len(data))
            data = torch.cat((data, prev_data.detach()), dim=0)
            target = torch.cat((target, prev_target.detach()), dim=0)
            del prev_data, prev_target

        # -----------------------------------------------------------------
        #
        # Generate data for current task

        if encoder is not None:
            mean, log_var = encoder(data)
            fake_data, _target = generator(target, mean=mean, log_var=log_var)
        else:
            fake_data, _target = generator(target)

        # -----------------------------------------------------------------
        #
        # We erase gradients in both generator and encoder. If any
        # loss function is used, optimizer_generator will be True
        # after finishing processing students.

        self.prof_optimizer.zero_grad()
        optimize_generator = False

        # -----------------------------------------------------------------
        #
        # Before doing stuff with the fake data, normalize the
        # posterior. This will put some gradients in the encoder.

        if encoder is not None:
            code = (mean, log_var)
            kld, recon_loss = self._do_vae(code, data, fake_data, nrmlz)

            if torch.is_tensor(recon_loss):
                info["Reconstruction"] = recon_loss.item() * nstudents
            info["KL encoder"] = kld.item() * nstudents

            (recon_loss + kld).backward(retain_graph=True)
            optimize_generator = True

            del code, kld, recon_loss

        for student in self.students:

            # -----------------------------------------------------------------
            #
            # First compute the student's predictions for both real
            # and fake data. Also, compute negative log likelihood
            # with target.

            real_output = student(data)
            fake_output = student(fake_data)
            real_nlls = F.cross_entropy(real_output, target, reduction="none")
            fake_nlls = F.cross_entropy(fake_output, target, reduction="none")

            # -----------------------------------------------------------------
            #
            # Compute negative log likelihood for student. It will be
            # backpropagated later as the professor' loss will put
            # some values into student's gradient. So, we will first
            # backpropagate the professor loss, clean the student's
            # gradients and then backpropagate this nll.

            student_loss = real_nlls.mean()
            if coeffs.c_l2 > 0:
                l2_loss = l2(student.parameters()) * coeffs.c_l2
                student_loss = student_loss + l2_loss
                del l2_loss
            student_losses.append(student_loss.item())

            # -----------------------------------------------------------------
            #
            # Compute gradients w.r.t. student's parameters for both
            # fake, and real examples.

            if self.need_grad:
                aligned_grads = self._get_aligned_grads(student, real_nlls,
                                                        fake_nlls, target)
            # -----------------------------------------------------------------
            #
            # Start computing losses for the professor. Accumulate them in
            # professor_loss. At the end we just perform backward once for
            # professor_loss.

            professor_loss = 0

            # -----------------------------------------------------------------
            #
            # Mean squared errors between NLLs on real and fake
            # examples.

            if coeffs.c_nll > 0:
                nll_mse = F.mse_loss(fake_nlls, real_nlls.detach())
                nll_mse *= coeffs.c_nll
                professor_loss += nll_mse
                info["NLL"] = info.get("NLL", 0) + nll_mse.item()
                del nll_mse

            # -----------------------------------------------------------------
            #
            # KL divergence between fake and real outputs.

            if coeffs.c_kl > 0:
                fake_logp = F.log_softmax(fake_output, dim=1)
                real_p = F.softmax(real_output, dim=1).detach()
                kldiv = F.kl_div(fake_logp, real_p) * coeffs.c_kl
                professor_loss += kldiv
                info["KL div"] = info.get("KL div", 0) + kldiv.item()
                del fake_logp, real_p, kldiv

            # -----------------------------------------------------------------
            #
            # Mean squared error between fake and real gradients.

            if coeffs.c_grad_mse > 0:
                grad_mse = 0
                for real_g, fake_g, _mask in aligned_grads:
                    grad_mse += mse(fake_g, real_g)
                grad_mse *= coeffs.c_grad_mse / len(aligned_grads)
                professor_loss += grad_mse
                info["Grad MSE"] = info.get("Grad MSE", 0) + grad_mse.item()
                del grad_mse

            # -----------------------------------------------------------------
            #
            # Cosine distance between fake and real gradients.

            if coeffs.c_grad_cos > 0:
                cos_loss = 0
                for real_grads, fake_grads, _mask in aligned_grads:
                    cos_loss += cos(fake_grads, real_grads)
                cos_loss *= coeffs.c_grad_cos / len(aligned_grads)
                professor_loss += cos_loss
                info["Grad Cos"] = info.get("Grad Cos", 0) + cos_loss.item()
                del cos_loss

            # -----------------------------------------------------------------
            #
            # Cross entropy for the student optimized with proposed
            # gradients.

            if coeffs.c_next_nll > 0:
                next_nll = self._next_nll(student, data, target, aligned_grads)
                professor_loss += next_nll
                info["Next NLL"] = info.get("Next NLL", 0) + next_nll.item()
                del next_nll

            # -----------------------------------------------------------------
            #
            # KL divergence between outputs of displaced parameters
            # and current outputs (both on real data). (Stefan's idea,
            # no clue why it's important.

            if coeffs.c_next_kl > 0:
                next_kl = self._next_kldiv(student, real_output, data,
                                           aligned_grads)
                professor_loss += next_kl
                info["Next KLdiv"] = info.get("Next KLdiv", 0) + next_kl.item()
                del next_kl

            # -----------------------------------------------------------------
            #
            # Mean Squared Error between the products of the Hessians
            # with some random vector.

            if coeffs.c_hess > 0:
                hess_loss = self._hess_vec(student, aligned_grads)
                professor_loss += hess_loss
                info["Hv MSE"] = info.get("Hv MSE", 0) + hess_loss.item()
                del hess_loss  # Avoid bugs

            # -----------------------------------------------------------------
            #
            # How bad is our current generator at fooling the
            # discriminator that the outputs of the student given fake
            # inputs result from real data.

            if coeffs.c_d > 0:
                d_loss, d_info = self._do_gan(real_output, fake_output, target)
                professor_loss += d_loss
                for key, value in d_info.items():
                    info[key] = info.get(key, 0) + value
                del d_loss, d_info  # Avoid bugs

            # -----------------------------------------------------------------
            #
            # Backward now so we won't keep the computational graph
            # through current student. First backpropagate from
            # professor's loss. This will accumulate gradients in both
            # generator's and encoder's parameters, but also in
            # current student's parameters. Therefore we need to erase
            # the student's gradients before backpropagation from its
            # loss.

            if torch.is_tensor(professor_loss):
                if torch.isnan(professor_loss).any().item():
                    return None, None  # Exit if parameters are garbage
                optimize_generator = True
                professor_loss /= nstudents
                professor_loss.backward(retain_graph=True)

            student_loss.backward()

            del professor_loss, student_loss

            # Backpropagation ended for current student.
            #
            # -----------------------------------------------------------------

        # -- If there is some loss, improve teacher

        if optimize_generator:
            if self.debug:
                generator_info = grad_info(generator)
                print(tabulate(generator_info))

                if encoder is not None:
                    encoder_info = grad_info(encoder)
                    print(tabulate(encoder_info))

            for key in info.keys():
                info[key] /= nstudents

            self.prof_optimizer.step()

        return student_losses, info

    def _next_kldiv(self, student, real_output, data, aligned_grads):
        next_kldiv = 0
        coeffs = self.coeffs
        for _real_grads, fake_grads, mask in aligned_grads:
            next_params = OrderedDict({})
            pg_pairs = zip(student.named_parameters(), fake_grads)
            for (name, param), grad in pg_pairs:
                next_params[name] = param.detach() - coeffs.next_lr * grad

            if mask is None:
                next_output = student(data, params=next_params)
                real_p = F.softmax(real_output, dim=1).detach()
            else:
                next_output = student(data[mask], params=next_params)
                real_p = F.softmax(real_output[mask], dim=1).detach()

            next_logp = F.log_softmax(next_output, dim=1)
            next_kldiv += F.kl_div(next_logp, real_p)

        next_kldiv *= coeffs.c_next_kl
        return next_kldiv

    def _hess_vec(self, student, aligned_grads):
        hess_loss = 0
        for real_grads, fake_grads, _mask in aligned_grads:
            rand_v = [torch.bernoulli(torch.rand_like(g)) for g in real_grads]
            real_hv = grad_of(real_grads, student.parameters(), rand_v)
            fake_hv = grad_of(fake_grads, student.parameters(), rand_v)
            hess_loss += mse(fake_hv, real_hv)
        hess_loss *= self.coeffs.c_hess / len(aligned_grads)
        return hess_loss

    def _next_nll(self, student, data, target, aligned_grads):
        next_nll = 0
        coeffs = self.coeffs
        for _real_grads, fake_grads, mask in aligned_grads:
            new_params = OrderedDict({})
            pg_pairs = zip(student.named_parameters(), fake_grads)
            for (name, param), grad in pg_pairs:
                new_params[name] = param.detach() - coeffs.next_lr * grad
            if mask is None:
                next_output = student(data, params=new_params)
                next_nll += F.cross_entropy(next_output, target)
            else:
                next_output = student(data[mask], params=new_params)
                next_nll += F.cross_entropy(next_output, target[mask])

        next_nll *= coeffs.c_next_nll / len(aligned_grads)
        return next_nll

    def _get_aligned_grads(self, student, real_nlls, fake_nlls, target):
        aligned_grads = []
        if self.grad_type == "batch":
            real_g = grad_of(real_nlls.mean(), student.parameters())
            fake_g = grad_of(fake_nlls.mean(), student.parameters())
            aligned_grads.append((real_g, fake_g, None))
        elif self.grad_type == "class":
            for class_idx in range(self.nclasses):
                mask = (target == class_idx)
                if mask.any().item():
                    real_nll_i = real_nlls[mask].mean()
                    fake_nll_i = fake_nlls[mask].mean()
                    real_g = grad_of(real_nll_i, student.parameters())
                    fake_g = grad_of(fake_nll_i, student.parameters())
                    aligned_grads.append((real_g, fake_g, mask))
        else:
            for idx in range(len(target)):
                real_g = grad_of(real_nlls[idx:idx + 1], student.parameters())
                fake_g = grad_of(fake_nlls[idx:idx + 1], student.parameters())
                aligned_grads.append((real_g, fake_g, slice(idx, idx+1)))
        return aligned_grads

    def _do_gan(self, real_output, fake_output, target):

        label = self.classeye[target]
        ones = torch.ones(target.size(), device=target.device)
        ones.unsqueeze_(1)
        zeros = torch.zeros(target.size(), device=target.device)
        zeros.unsqueeze_(1)

        # Improve discriminator

        self.d_optimizer.zero_grad()

        d_real_in = torch.cat((real_output.detach(), label), dim=1)
        d_real_out = self.discriminator(d_real_in)
        loss_real = self.bce_loss(d_real_out, ones)

        d_fake_in = torch.cat((fake_output.detach(), label), dim=1)
        d_fake_out = self.discriminator(d_fake_in)
        loss_fake = self.bce_loss(d_fake_out, zeros)

        (loss_real + loss_fake).backward()
        self.d_optimizer.step()

        # Improve generator

        d_gen_in = torch.cat((fake_output, label), dim=1)
        d_gen_out = self.discriminator(d_gen_in)
        d_gen_loss = self.bce_loss(d_gen_out, ones)

        # Return useful info

        info = dict({})
        info["Discriminator BCE - gen "] = d_gen_loss.item()
        info["Discriminator avg - gen "] = d_gen_out.mean().item()
        info["Discriminator BCE - real"] = loss_real.item()
        info["Discriminator avg - real"] = d_real_out.mean().item()
        info["Discriminator BCE - fake"] = loss_fake.item()
        info["Discriminator avg - fake"] = d_fake_out.mean().item()

        return d_gen_loss * self.coeffs.c_d, info

    def _do_vae(self, code, data, fake_data, nrmlz):
        mean, log_var = code
        if self.coeffs.c_recon > 0:
            data_mean, data_std = nrmlz
            batch_size = len(data)
            with torch.no_grad():
                scaled_data = data_mean + data * data_std
                scaled_data.clamp_(min=0, max=1)
            recon_loss = F.binary_cross_entropy(
                (fake_data.view(batch_size, -1) + 1) / 2,
                scaled_data.view(batch_size, -1).detach(),
                reduction='sum') * self.coeffs.c_recon
        else:
            recon_loss = 0
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        kld *= self.coeffs.c_latent_kl

        return kld, recon_loss
