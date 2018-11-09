from itertools import chain
from collections import OrderedDict
import os.path
from argparse import Namespace
from tabulate import tabulate
from termcolor import colored as clr
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from models import Student
from models import get_model
from models import generative, classifiers
from models.classifiers import sample_classifier
from professors.professor import Professor

from utils import get_optimizer, nparams, grad_info
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


def what_to_reset(ref_acc: float, nclasses, strategy, accs, ends):
    to_reset = []
    start_idx, end_idx = ends
    nstudents = len(accs[start_idx:end_idx])

    if ref_acc > 150. / nclasses:
        if strategy == "powspace":
            thrs = np.power(np.linspace(np.power((150. / nclasses), np.e),
                                        np.power(ref_acc, np.e),
                                        nstudents),
                            1/np.e)
        else:
            thrs = np.linspace((150. / nclasses),
                               (ref_acc),
                               nstudents)
        thrs = thrs[1:]
        balance = 0
        prev_idxs = []
        for thr in thrs[::-1]:
            good_idxs = []
            for sidx in range(start_idx, end_idx):
                if sidx not in prev_idxs and accs[sidx] > thr:
                    good_idxs.append(sidx)
            np.random.shuffle(good_idxs)
            balance += len(good_idxs) - 1
            if balance > 0:
                to_reset.append(np.random.choice(good_idxs))
                balance -= 1
            while balance > 0:
                good_idxs.pop()
                balance -= 1
            prev_idxs.extend(good_idxs)

    return to_reset


class GenerativeProfessor(Professor):

    def __init__(self, args: Namespace, device, start_params=None) -> None:
        super(GenerativeProfessor, self).__init__("GENE-PROF", args.verbose)
        self.args = args
        self.nclasses = nclasses = args.nclasses
        self.in_size = in_size = args.in_size
        self.nrmlz = args.nrmlz
        self.crt_device = device
        self.start_params = start_params

        self._init_students()
        self.nstudents = nstudents = len(self.students)
        self.generator_idx = 0

        self.generator, self.encoder, self.discriminator = None, None, None
        self.d_optimizer, self.prof_optimizer = None, None
        self.old_generator = None

        self.classeye = torch.eye(args.nclasses, device=device)

        self.trained_on_fake = args.trained_on_fake
        if args.trained_on_fake > len(self.students):
            raise ValueError("Trained on fake more than existing.")

        # ----------------------------------------------------------------------

        self.label_to_discriminator = args.label_to_discriminator
        self.permute_before_discriminator = args.permute_before_discriminator

        self._create_components()

        self.coeffs = coeffs = Namespace()

        self.coeffs.c_nll = args.c_nll
        self.coeffs.c_kl = args.c_kl
        self.coeffs.c_contrast_kl = args.c_contrast_kl
        self.coeffs.c_grad_mse = args.c_grad_mse
        self.coeffs.c_grad_cos = args.c_grad_cos
        self.coeffs.c_next_nll = args.c_next_nll
        self.coeffs.c_contrast_next_nll = args.c_contrast_next_nll
        self.coeffs.c_next_nll2 = args.c_next_nll2
        self.coeffs.c_next_kl = args.c_next_kl
        self.coeffs.c_hess = args.c_hess
        self.coeffs.c_d = args.c_d
        self.coeffs.c_recon = args.c_recon
        self.coeffs.c_l2 = args.c_l2
        self.coeffs.c_latent_kl = args.c_latent_kl
        self.coeffs.next_lr = args.next_lr
        self.coeffs.target_dropout = args.target_dropout

        # ---------------------------------------------------------------------
        #
        # Let's check what needs to be computed (i.e. contrast data,
        # gradinets)

        def check_need(lst):
            return any(getattr(coeffs, n) > 0 for n in lst)

        w_contrast = ["c_contrast_kl", "c_contrast_next_nll"]
        self.need_contrast = check_need(w_contrast)
        self.info("Contrast data will be generated.")

        w_real_grad = ["c_grad_mse", "c_grad_cos", "c_next_nll2", "c_hess"]
        self.need_real_grad = check_need(w_real_grad)

        w_fake_grad = ["c_grad_mse", "c_grad_cos", "c_next_nll", "c_next_kl",
                       "c_hess"]
        self.need_fake_grad = check_need(w_fake_grad)

        w_contrast_grad = ["c_contrast_next_nll"]
        self.need_contrast_grad = check_need(w_contrast_grad)

        self.need_some_grad = self.need_fake_grad or self.need_real_grad or \
                              self.need_contrast_grad

        # ---------------------------------------------------------------------

        self.grad_type = args.grad_type
        assert self.grad_type in ["batch", "example", "class"]
        if self.grad_type == "example":
            self.grad_samples = args.grad_samples
        else:
            self.grad_samples = None

        self.eval_samples = args.eval_samples
        self.info(args.eval_samples, "samples will be used during teaching.")

        # ---------------------------------------------------------------------

        if isinstance(args.student_reset, list):
            if len(args.student_reset) == nstudents:
                self.student_reset = args.student_reset
            else:
                raise ValueError("Reset times must match no. of students.")
        elif isinstance(args.student_reset, int):
            self.student_reset = [args.student_reset] * nstudents
        elif args.student_reset in ["linspace", "powspace", "everystep"]:
            self.student_reset = args.student_reset
        else:
            raise ValueError("Expected int or list of ints or string. Got" +
                             str(args.student_reset))

        # ---------------------------------------------------------------------

        self.avg_fake_acc = [100.0 / nclasses] * len(self.students)
        self.avg_real_acc = [100.0 / nclasses] * len(self.students)
        self.max_known_real_acc = 150 /nclasses
        self.last_perf = None  # used during evaluation

        self.info_trace = OrderedDict({})
        self.nseen = 0
        self.epoch = 1
        self.report_freq = args.report_freq
        self.last_report = 0

    def _init_students(self):
        in_size, nclasses = self.in_size, self.nclasses
        self.students, self.student_optimizers = [], []
        for idx in range(self.args.nstudents):
            if not self.args.random_students:
                student = get_model(classifiers, self.args.student,
                                    in_size=in_size,
                                    nclasses=nclasses).to(self.crt_device)
                if self.start_params:
                    student.load_state_dict(self.start_params)
            else:
                student = sample_classifier(in_size, nclasses).to(self.crt_device)
            student_optimizer = get_optimizer(student.parameters(),
                                              self.args.student_optimizer)
            self.students.append(student)
            self.student_optimizers.append(student_optimizer)
            self.info(nparams(student, name="Student model"))

    def _create_components(self):
        args = self.args
        self.generator = get_model(generative,
                                   args.generator,
                                   in_size=self.in_size,
                                   nclasses=self.nclasses,
                                   nz=args.nz,
                                   nperf=args.nperf)
        self.generator.to(self.crt_device)
        self.info(nparams(self.generator,
                          name=f"Generator:{self.generator_idx:d}"))

        if hasattr(args, "encoder") and args.c_latent_kl > 0:
            self.encoder = get_model(generative, args.encoder,
                                     in_size=args.in_size, nz=args.nz)
            self.encoder.to(self.crt_device)
            all_params = chain(self.encoder.parameters(),
                               self.generator.parameters())
            self.info(nparams(self.encoder, name=f"Encoder"))
        else:
            all_params = self.generator.parameters()

        self.prof_optimizer = get_optimizer(all_params, args.optimizer)

        if hasattr(args, "discriminator") and args.c_d > 0:
            discriminator = get_model(generative, args.discriminator,
                                      nclasses=args.nclasses,
                                      use_labels=self.label_to_discriminator)
            discriminator.to(self.crt_device)
            self.discriminator = discriminator
            self.bce_loss = nn.BCELoss()
            self.d_optimizer = optim.Adam(discriminator.parameters(), lr=.001)

            self.info(nparams(self.encoder, name=f"Discriminator"))

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

    def eval_student(self, student: Student,
                     step: int,
                     nsamples: int = None) -> Tensor:
        if step == 0:
            self.last_perf = last_perf = 1 / self.nclasses
        else:
            last_perf = self.last_perf
        if nsamples is None:
            nsamples = self.eval_samples
        with torch.no_grad():
            data, target = self.generator(nsamples=nsamples, perf=last_perf)

        output = student(data)

        with torch.no_grad():
            pred = output.max(1, keepdim=True)[1]
            correct = pred.eq(target.view_as(pred)).sum().item()
        self.last_perf = last_perf = (correct / len(data) * 100)
        return F.cross_entropy(output, target), last_perf

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
            fake_data, _ = self.generator(nsamples=64,
                                          perf=torch.linspace(10, 90, 64))
            save_image(fake_data.cpu(),
                       os.path.join(out_path, f"samples_{epoch_no:04d}.png"))

        # 3. save some comparisons

        with torch.no_grad():
            mean, log_var = (None, None) if encoder is None else encoder(data)
            fake_data, _target = generator(target, mean=mean, log_var=log_var,
                                           perf=torch.linspace(10, 90, len(data)))
            all_data = torch.cat((data, fake_data), dim=0).cpu()
            save_image(all_data,
                       os.path.join(out_path, f"recons_{epoch_no:04d}.png"))

    """
    def end_task(self, is_last: bool = False) -> None:
        if is_last:
            return
        self.old_generator = self.generator
        self.old_generator.eval()

        self.generator, self.encoder, self.discriminator = None, None, None
        self.prof_optimizer, self.d_optimizer = None, None
        self.generator_idx += 1

        self._create_components()


    def init_student(self, student, optimizer_args, nsteps: int = None):
        if nsteps is None:
            nsteps = np.random.randint(1000)
        nsamples = self.eval_samples
        student_optimizer = get_optimizer(student.parameters(),
                                          optimizer_args)
        for _step in range(nsteps):
            student_optimizer.zero_grad()
            with torch.no_grad():
                data, target = self.generator(nsamples=nsamples)
            output = student(data)
            loss = F.cross_entropy(output, target)
            if self.coeffs.c_l2 > 0:
                l2_loss = l2(student.parameters()) * self.coeffs.c_l2
                loss = loss + l2_loss
            loss.backward()
            student_optimizer.step()

        with torch.no_grad():
            data, target = self.generator(nsamples=nsamples)
            output = student(data)
            loss = F.cross_entropy(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct = pred.eq(target.view_as(pred)).sum().item()
            acc = (correct / len(data) * 100)
            print(f"[INIT_] Left student after {nsteps:d} steps "
                  f"with Synthetic NLL={loss:.3f} and "
                  f"Accuracy: {acc:.2f}%")
        student.zero_grad()
    """

    def process(self, data, target):
        student_losses = []
        info = OrderedDict({})
        info_max = OrderedDict({})
        coeffs = self.coeffs
        encoder = self.encoder
        generator = self.generator

        orig_data, orig_target = data, target
        # ---------------------------------------------------------------------
        #
        # If gradients are computed per example, we'll take a fixed
        # number of samples for each student.

        if self.grad_type == "example" and self.need_some_grad:
            ngrad_samples = min(len(data), self.grad_samples)

        nstudents = len(self.students)  # type: int

        for sidx, student in enumerate(self.students):

            perf = self.avg_fake_acc[sidx]

            # -----------------------------------------------------------------
            #
            # If there is an old generator, we just add some synthetic
            # data for previous tasks. The batch size gets doubled.

            if self.old_generator is not None:
                with torch.no_grad():
                    prev_data, prev_target = self.old_generator(
                        nsamples=len(data), perf=perf)
                data = torch.cat((orig_data, prev_data.detach()), dim=0)
                target = torch.cat((orig_target, prev_target.detach()), dim=0)
                del prev_data, prev_target

            # -----------------------------------------------------------------
            #
            # Generate data for current task

            if coeffs.target_dropout > 0:
                tmask = torch.bernoulli(torch.full(target.size(),
                                                   coeffs.target_dropout))
                tmask = tmask.to(target.device)
            else:
                tmask = None

            if encoder is not None:
                mean, log_var = encoder(data)
                fake_data, _target = generator(target, mean=mean,
                                               log_var=log_var,
                                               tmask=tmask,
                                               perf=perf)
            else:
                fake_data, _target = generator(target, tmask=tmask, perf=perf)

            if self.need_contrast:
                with torch.no_grad():
                    contrast_data, _target = generator(target, tmask=tmask,
                                                       perf=perf)
            else:
                contrast_data = None

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
                kld, recon_loss = self._do_vae(code, data, fake_data)

                if torch.is_tensor(recon_loss):
                    info["Reconstruction"] = recon_loss.item()
                info["KL encoder"] = kld.item()

                (recon_loss + kld).backward(retain_graph=True)
                optimize_generator = True

                del code, kld, recon_loss

            # -----------------------------------------------------------------
            #
            # First compute the student's predictions for both real
            # and fake data. Also, compute negative log likelihood
            # with target.

            real_output = student(data)
            fake_output = student(fake_data)
            real_nlls = F.cross_entropy(real_output, target, reduction="none")
            fake_nlls = F.cross_entropy(fake_output, target, reduction="none")

            _, real_pred = real_output.max(1)
            real_acc = real_pred.eq(target).sum().item() / len(data) * 100

            self.max_known_real_acc = max(real_acc, self.max_known_real_acc)
            info_max["Max. Known Real Acc"] = self.max_known_real_acc

            _, fake_pred = fake_output.max(1)
            fake_acc = fake_pred.eq(target).sum().item() / len(data) * 100

            if self.need_contrast:
                if self.need_contrast_grad:
                    contrast_output = student(contrast_data)
                    contrast_nlls = F.cross_entropy(contrast_output, target,
                                                    reduction="none")
                else:
                    with torch.no_grad():
                        contrast_output = student(contrast_data)
                        contrast_nlls = None

            # -----------------------------------------------------------------
            #
            # Compute negative log likelihood for student. It will be
            # backpropagated later as the professor' loss will put
            # some values into student's gradient. So, we will first
            # backpropagate the professor loss, clean the student's
            # gradients and then backpropagate this nll.

            if sidx < nstudents - self.trained_on_fake:
                student_loss = real_nlls.mean()
            else:
                student_loss = fake_nlls.mean()

            if coeffs.c_l2 > 0:
                l2_loss = l2(student.parameters()) * coeffs.c_l2
                student_loss = student_loss + l2_loss
                del l2_loss
            student_losses.append(student_loss.item())

            # -----------------------------------------------------------------
            #
            # Compute gradients w.r.t. student's parameters for both
            # fake, and real examples.

            if self.need_some_grad:
                if self.grad_type == "example":
                    idxs = [i for i in range(ngrad_samples)
                            if i % nstudents == sidx]
                else:
                    idxs = None
                aligned_grads = self._get_aligned_grads(student,
                                                        real_nlls, fake_nlls,
                                                        contrast_nlls,
                                                        target, idxs=idxs)
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

                if coeffs.c_contrast_kl > 0:
                    with torch.no_grad():
                        contrast_p = F.softmax(contrast_output, dim=1)
                    contrast_kldiv = F.kl_div(fake_logp, contrast_p)
                    contrast_kldiv *= coeffs.c_kl * coeffs.c_contrast_kl
                    professor_loss -= contrast_kldiv
                    info["KL div - contr"] = info.get("KL div - contr", 0) +\
                        contrast_kldiv.item()

                del fake_logp, real_p, kldiv

            # -----------------------------------------------------------------
            #
            # Mean squared error between fake and real gradients.

            if coeffs.c_grad_mse > 0:
                grad_mse = 0
                for real_g, fake_g, _contrast_g, _mask in aligned_grads:
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
                for real_grads, fake_grads, _contrast_g, _mask in aligned_grads:
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
                next_l = self._next_nll(student, data, target, aligned_grads)
                next_nll, contrast_next_nll = next_l
                professor_loss += next_nll
                info["Next NLL"] = info.get("Next NLL", 0) + next_nll.item()
                if contrast_next_nll is not None:
                    professor_loss -= contrast_next_nll
                    info["Next NLL - contr"] = info.get("Next NLL - contr", 0) +\
                        contrast_next_nll.item()
                del next_nll

            # -----------------------------------------------------------------
            #
            # Cross entropy for the student optimized with proposed
            # gradients. (symmetric: gradients from real data applied
            # on synthetic data)

            if coeffs.c_next_nll2 > 0:
                next_nll2 = self._next_nll2(student, fake_data, target, aligned_grads)
                professor_loss += next_nll2
                info["Next NLL (2)"] = info.get("Next NLL (2)", 0) + next_nll2.item()
                del next_nll2

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
                    return True
                optimize_generator = True
                # professor_loss /= nstudents
                professor_loss.backward(retain_graph=True)

            # -- If there is some loss, improve teacher

            if optimize_generator:
                if self.verbose > 1:
                    generator_info = grad_info(generator)
                    print(tabulate(generator_info))

                    if encoder is not None:
                        encoder_info = grad_info(encoder)
                        print(tabulate(encoder_info))

                self.prof_optimizer.step()

            student.zero_grad()
            if sidx < nstudents - self.trained_on_fake:
                student_loss.backward()
            else:
                student_loss.backward(retain_graph=True)

            self.student_optimizers[sidx].step()

            del professor_loss, student_loss

            # Backpropagation ended for current student.
            #
            # -----------------------------------------------------------------


            # -----------------------------------------------------------------
            #
            # -- Post computation

            self.avg_fake_acc[sidx] *= .75
            self.avg_fake_acc[sidx] += .25 * fake_acc
            self.avg_real_acc[sidx] *= .75
            self.avg_real_acc[sidx] += .25 * real_acc

        self.reset_students(len(orig_data))

        for key, value in info.items():
            self.info_trace.setdefault(key, []).append(value / nstudents)

        for key, value in info_max.items():
            [old_value] = self.info_trace.get(key, [value])
            self.info_trace[key] = [max(old_value, value)]

        self.nseen = nseen = self.nseen + len(orig_data)
        self.report()
        return False

    def report(self):
        if self.nseen - self.last_report >= self.report_freq:
            t = [(key, np.mean(vals)) for (key, vals) in self.info_trace.items()]
            t = [("Epoch", self.epoch), ("Seen", self.nseen)] + t
            print(tabulate(t))
            self.info_trace.clear()
            self.last_report += self.report_freq

            avg_fake_acc, avg_real_acc = self.avg_fake_acc, self.avg_real_acc
            fake_accs = [f"{acc:5.2f}" for acc in avg_fake_acc]
            real_accs = [clr(f"{acc:5.2f}", "yellow") for acc in avg_real_acc]

            nreal = self.nstudents - self.trained_on_fake
            if nreal > 0:
                max_r_on_f_idx = np.argmax(avg_fake_acc[:nreal])
                max_r_on_f = avg_fake_acc[max_r_on_f_idx]
                fake_accs[max_r_on_f_idx] = clr(f"{max_r_on_f:5.2f}",
                                                "white", "on_cyan")

                max_r_on_r_idx = np.argmax(avg_real_acc[:nreal])
                max_r_on_r = avg_real_acc[max_r_on_r_idx]
                real_accs[max_r_on_r_idx] = clr(f"{max_r_on_r:5.2f}",
                                                "yellow", "on_cyan")

            if self.trained_on_fake > 0:
                max_f_on_f_idx = np.argmax(avg_fake_acc[nreal:])
                max_f_on_f = avg_fake_acc[max_f_on_f_idx + nreal]
                fake_accs[max_f_on_f_idx + nreal] = clr(f"{max_f_on_f:5.2f}",
                                                "white", "on_green")

                max_f_on_r_idx = np.argmax(avg_real_acc[nreal:])
                max_f_on_r = avg_real_acc[max_f_on_r_idx + nreal]
                real_accs[max_f_on_r_idx + nreal] = clr(f"{max_f_on_r:5.2f}",
                                                "yellow", "on_green")

            self.info(" | ".join(fake_accs[:nreal]),
                      clr("|||", "yellow"),
                      " | ".join(fake_accs[nreal:]),
                      tags=["@FAKE"])
            self.info(" | ".join(real_accs[:nreal]),
                      clr("|||", "yellow"),
                      " | ".join(real_accs[nreal:]),
                      tags=["@REAL"])
            self.info("----")

    def _next_kldiv(self, student, real_output, data, aligned_grads):
        next_kldiv = 0
        coeffs = self.coeffs
        for _real_grads, fake_grads, _contrast_grads, mask in aligned_grads:
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
        for real_grads, fake_grads, _contrast_grads, _mask in aligned_grads:
            rand_v = [torch.bernoulli(torch.rand_like(g)) for g in real_grads]
            real_hv = grad_of(real_grads, student.parameters(), rand_v)
            fake_hv = grad_of(fake_grads, student.parameters(), rand_v)
            hess_loss += mse(fake_hv, real_hv)
        hess_loss *= self.coeffs.c_hess / len(aligned_grads)
        return hess_loss

    def _next_nll(self, student, data, target, aligned_grads):
        next_nll = 0
        coeffs = self.coeffs
        do_contrast = coeffs.c_contrast_next_nll > 0
        contrast_nll = 0 if do_contrast else None

        for _real_grads, fake_grads, contrast_grads, mask in aligned_grads:
            new_params, contrast_params = OrderedDict({}), OrderedDict({})
            pg_pairs = zip(student.named_parameters(), fake_grads, contrast_grads)
            for (name, param), grad, contrast_grad in pg_pairs:
                new_params[name] = param.detach() - coeffs.next_lr * grad
                if do_contrast:
                    contrast_params[name] = param.detach() -\
                        coeffs.next_lr * contrast_grad
            if mask is None:
                next_output = student(data, params=new_params)
                next_nll += F.cross_entropy(next_output, target)
                if do_contrast:
                    with torch.no_grad():
                        contrast_output = student(data, params=contrast_params)
                        contrast_p = F.softmax(contrast_output, dim=1)
                    next_logp = F.log_softmax(next_output, dim=1)
                    contrast_nll += F.kl_div(next_logp, contrast_p)

            else:
                next_output = student(data[mask], params=new_params)
                next_nll += F.cross_entropy(next_output, target[mask])
                if do_contrast:
                    with torch.no_grad():
                        contrast_output = student(data[mask],
                                                  params=contrast_params)
                        contrast_p = F.softmax(contrast_output, dim=1)
                    next_logp = F.log_softmax(next_output, dim=1)
                    contrast_nll += F.kl_div(next_logp, contrast_p)

        next_nll *= coeffs.c_next_nll / len(aligned_grads)
        if do_contrast:
            contrast_nll *= coeffs.c_next_nll * coeffs.c_contrast_next_nll
            contrast_nll /= len(aligned_grads)
        return next_nll, contrast_nll

    def _next_nll2(self, student, fake_data, target, aligned_grads):
        next_nll2 = 0
        coeffs = self.coeffs
        for real_grads, _fake_grads, _contrast_grads, mask in aligned_grads:
            new_params = OrderedDict({})
            pg_pairs = zip(student.named_parameters(), real_grads)
            for (name, param), grad in pg_pairs:
                new_params[name] = param.detach() - coeffs.next_lr * grad.detach()
            if mask is None:
                next_output = student(fake_data, params=new_params)
                next_nll2 += F.cross_entropy(next_output, target)
            else:
                next_output = student(fake_data[mask], params=new_params)
                next_nll2 += F.cross_entropy(next_output, target[mask])

        next_nll2 *= coeffs.c_next_nll2 / len(aligned_grads)
        return next_nll2

    def _get_aligned_grads(self, student,
                           real_nlls, fake_nlls, contrast_nlls,
                           target, idxs=None):
        aligned_grads = []
        if self.grad_type == "batch":
            real_g, fake_g, contrast_g = None, None, None
            if self.need_real_grad:
                real_g = grad_of(real_nlls.mean(), student.parameters())
            if self.need_fake_grad:
                fake_g = grad_of(fake_nlls.mean(), student.parameters())
            if self.need_contrast_grad:
                contrast_g = grad_of(contrast_nlls.mean(), student.parameters())
                for cgrad in contrast_g:
                    cgrad.detach_()
            aligned_grads.append((real_g, fake_g, contrast_g, None))
        elif self.grad_type == "class":
            for class_idx in range(self.nclasses):
                mask = (target == class_idx)
                if mask.any().item():
                    real_g, fake_g, contrast_g = None, None, None
                    if self.need_real_grad:
                        real_nll_i = real_nlls[mask].mean()
                        real_g = grad_of(real_nll_i, student.parameters())
                    if self.need_fake_grad:
                        fake_nll_i = fake_nlls[mask].mean()
                        fake_g = grad_of(fake_nll_i, student.parameters())
                    if self.need_contrast_grad:
                        contrast_nll_i = contrast_nlls[mask].mean()
                        contrast_g = grad_of(contrast_nll_i, student.parameters())
                        contrast_g = tuple([cg.detach() for cg in contrast_g])

                    aligned_grads.append((real_g, fake_g, contrast_g, mask))
        else:
            if idxs is None:
                idxs = range(len(target))
            for idx in idxs:
                real_g, fake_g, contrast_g = None, None, None
                if self.need_real_grad:
                    real_g = grad_of(real_nlls[idx:idx + 1], student.parameters())
                if self.need_fake_grad:
                    fake_g = grad_of(fake_nlls[idx:idx + 1], student.parameters())
                if self.need_contrast_grad:
                    contrast_g = grad_of(contrast_nlls[idx:idx + 1], student.parameters())
                    contrast_g = [cg.detach() for cg in contrast_g]
                aligned_grads.append((real_g, fake_g, contrast_g, slice(idx, idx+1)))
        return aligned_grads

    def _do_gan(self, real_output, fake_output, target):

        label = self.classeye[target]
        ones = torch.ones(target.size(), device=target.device)
        ones.unsqueeze_(1)
        zeros = torch.zeros(target.size(), device=target.device)
        zeros.unsqueeze_(1)

        # Improve discriminator

        self.d_optimizer.zero_grad()

        if self.permute_before_discriminator:
            with torch.no_grad():
                perm = torch.randperm(self.nclasses).long().to(real_output.device)
                perm_real = real_output.index_select(1, perm)
                perm_fake = fake_output.index_select(1, perm)
        else:
            perm_real = real_output.detach()
            perm_fake = fake_output.detach()

        if self.label_to_discriminator:
            with torch.no_grad():
                d_real_in = torch.cat((perm_real, label), dim=1)
                d_fake_in = torch.cat((perm_fake, label), dim=1)
        else:
            d_real_in = perm_real
            d_fake_in = perm_fake

        d_real_out = self.discriminator(d_real_in)
        loss_real = self.bce_loss(d_real_out, ones)

        d_fake_out = self.discriminator(d_fake_in)
        loss_fake = self.bce_loss(d_fake_out, zeros)

        (loss_real + loss_fake).backward()
        self.d_optimizer.step()

        # Improve generator

        if self.label_to_discriminator:
            d_gen_in = torch.cat((fake_output, label), dim=1)
        else:
            d_gen_in = fake_output

        d_gen_out = self.discriminator(d_gen_in)
        d_gen_loss = self.bce_loss(d_gen_out, ones)

        # Return useful info

        info = OrderedDict({})
        info["Discriminator BCE - gen "] = d_gen_loss.item()
        info["Discriminator BCE - real"] = loss_real.item()
        info["Discriminator BCE - fake"] = loss_fake.item()
        info["Discriminator avg - gen "] = d_gen_out.mean().item()
        info["Discriminator avg - real"] = d_real_out.mean().item()
        info["Discriminator avg - fake"] = d_fake_out.mean().item()

        return d_gen_loss * self.coeffs.c_d, info

    def _do_vae(self, code, data, fake_data):
        mean, log_var = code
        if self.coeffs.c_recon > 0:
            data_mean, data_std = self.nrmlz
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

    def reset_students(self, seen_samples: int):
        student_reset = self.student_reset
        nstudents = len(self.students)
        nreal = nstudents - self.trained_on_fake
        nfake = self.trained_on_fake
        in_size, nclasses = self.in_size, self.nclasses

        if student_reset == "everystep":
            to_reset = list(range(nstudents))
        elif isinstance(student_reset, str):
            ref_acc = self.max_known_real_acc

            to_reset_1, to_reset_2 = [], []
            if nreal > 0:
                to_reset_1 = what_to_reset(ref_acc, nclasses,
                                           student_reset,
                                           self.avg_real_acc,
                                           (0, nreal))
            if nfake > 0:
                to_reset_2 = what_to_reset(ref_acc, nclasses,
                                           student_reset,
                                           self.avg_fake_acc,
                                           (nreal,
                                            nstudents))
            to_reset = to_reset_1 + to_reset_2

        else:
            to_reset = []
            for sidx, freq in zip(range(0, nstudents), student_reset):
                p_reset = seen_samples / freq
                if np.random.sample() < p_reset:
                    to_reset.append(sidx)

        nreal = self.nstudents - self.trained_on_fake
        max_f_on_f_idx, max_f_on_r_idx = None, None
        if nreal > 0:
            max_f_on_r_idx = np.argmax(self.avg_fake_acc[:nreal])
        if self.trained_on_fake > 0:
            max_f_on_f_idx = nreal + np.argmax(self.avg_fake_acc[nreal:])

        if to_reset and student_reset != "everystep":
            colored = []
            for sidx, acc in enumerate(self.avg_fake_acc):
                if sidx in to_reset:
                    clrs = ("white", "on_magenta")
                elif sidx == max_f_on_r_idx:
                    clrs = ("white", "on_cyan")
                elif sidx == max_f_on_f_idx:
                    clrs = ("white", "on_green")
                else:
                    clrs = ("white",)
                colored.append(clr(f"{acc:5.2f}", *clrs))
            self.info(" | ".join(colored[:nreal]),
                      clr("|||", "yellow"),
                      " | ".join(colored[nreal:]),
                      tags=["RESET"])

        for sidx in to_reset:
            if self.start_params:
                self.students[sidx].load_state_dict(self.start_params)
            elif self.args.random_students:
                self.students[sidx] = sample_classifier(in_size, nclasses)
                self.students[sidx].to(self.crt_device)
            else:
                self.students[sidx].reset_weights()

            self.student_optimizers[sidx] = get_optimizer(
                self.students[sidx].parameters(),
                self.args.student_optimizer)
            self.avg_fake_acc[sidx] = 100. / self.nclasses
            self.avg_real_acc[sidx] = 100. / self.nclasses


    def end_epoch(self):
        self.epoch += 1