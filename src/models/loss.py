import torch
from apex import amp
from torch_utils import to_cuda


def check_overflow(grad):
    cpu_sum = float(grad.float().sum())
    if cpu_sum == float("inf") or cpu_sum == -float("inf") or cpu_sum != cpu_sum:
        return True
    return False

class WGANLoss:

    def __init__(self, discriminator, generator, opt_level):
        self.generator = generator
        self.discriminator = discriminator
        if opt_level == "O0":
            self.wgan_gp_scaler = amp.scaler.LossScaler(1)
        else:
            self.wgan_gp_scaler = amp.scaler.LossScaler(2**14)

    
    def update_optimizers(self, d_optimizer, g_optimizer):
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
    
    def compute_gradient_penalty(self, real_data, fake_data):
        epsilon_shape = [real_data.shape[0]] + [1]*(real_data.dim() - 1)
        epsilon = torch.rand(epsilon_shape)
        epsilon = to_cuda(epsilon)
        epsilon = epsilon.to(fake_data.dtype)
        real_data = real_data.to(fake_data.dtype)
        x_hat = epsilon * real_data + (1-epsilon) * fake_data.detach()
        x_hat.requires_grad = True
        logits = self.discriminator(x_hat)
        logits = logits.sum() * self.wgan_gp_scaler.loss_scale()
        grad = torch.autograd.grad(
            outputs=logits,
            inputs=x_hat,
            grad_outputs=torch.ones(logits.shape).to(fake_data.dtype).to(fake_data.device),
            create_graph=True
        )[0] 
        grad = grad.view(x_hat.shape[0], -1)
        if check_overflow(grad):
            print("Overflow in gradient penalty calculation.")
            self.wgan_gp_scaler._loss_scale /= 2
            print("Scaling down loss to:", self.wgan_gp_scaler._loss_scale)
            return None
        grad = grad / self.wgan_gp_scaler.loss_scale()
        gradient_pen = ((grad.norm(p=2, dim=1) - 1)**2)
        to_backward = gradient_pen.sum() * 10 
        with amp.scale_loss(to_backward, self.d_optimizer, loss_id=1) as scaled_loss:
            scaled_loss.backward(retain_graph=True)
        return gradient_pen.detach().mean()

    def step(self, real_data):
        # Train Discriminator
        z = self.generator.generate_latent_variable(real_data.shape[0])
        with torch.no_grad():
            fake_data = self.generator(z)
        real_scores = self.discriminator(real_data)
        fake_scores = self.discriminator(fake_data)
        # Wasserstein-1 Distance
        wasserstein_distance = (real_scores - fake_scores).squeeze()

        # Epsilon penalty
        epsilon_penalty = (real_scores ** 2).squeeze()

        self.d_optimizer.zero_grad()
        gradient_pen = self.compute_gradient_penalty(real_data, fake_data)
        if gradient_pen is None:
            return None

        to_backward1 = (- wasserstein_distance).sum()
        with amp.scale_loss(to_backward1, self.d_optimizer, loss_id=0) as scaled_loss:
            scaled_loss.backward(retain_graph=True)

        to_backward3 = epsilon_penalty.sum() * 0.001
        with amp.scale_loss(to_backward3, self.d_optimizer, loss_id=2) as scaled_loss:
            scaled_loss.backward()

        self.d_optimizer.step()
        z = self.generator.generate_latent_variable(real_data.shape[0])
        fake_data = self.generator(z)
        # Forward G
        for p in self.discriminator.parameters():
            p.requires_grad = False
        fake_scores = self.discriminator(fake_data)
        G_loss = (-fake_scores).sum()

        self.g_optimizer.zero_grad()
        with amp.scale_loss(G_loss, self.g_optimizer, loss_id=3) as scaled_loss:
            scaled_loss.backward()
        self.g_optimizer.step()
        for p in self.discriminator.parameters():
            p.requires_grad = True
        return wasserstein_distance.mean().detach(), gradient_pen.mean().detach(), real_scores.mean().detach(), fake_scores.mean().detach(), epsilon_penalty.mean().detach()
