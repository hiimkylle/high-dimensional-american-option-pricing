from config import *
import torch


class AGProcess:
    def __init__(self, a, alpha):
        # Do not rewrap with torch.tensor() to preserve gradient flow
        self.a = (
            a
            if isinstance(a, torch.Tensor)
            else torch.tensor(a, dtype=DT, device=device)
        )
        self.alpha = (
            alpha
            if isinstance(alpha, torch.Tensor)
            else torch.tensor(alpha, dtype=DT, device=device)
        )
        self.dim = self.alpha.numel()

    def simulate_increments(self, batch_size=1):
        # Vectorized version for speed and differentiability
        shape_g = delta_t * (1 - self.a * self.alpha) / self.alpha
        rate_g = 1 / self.alpha

        # Expand for broadcasting
        shape_g = shape_g[:, None, None]
        rate_g = rate_g[:, None, None]

        shape_g = torch.clamp(shape_g, min=1e-8)
        rate_g = torch.clamp(rate_g, min=1e-8)

        # Sample idiosyncratic increments
        g_increments = torch.distributions.Gamma(
            shape_g.expand(-1, batch_size, sample_size),
            rate_g.expand(-1, batch_size, sample_size),
        ).rsample()

        adt = self.a * delta_t
        adt = torch.clamp(adt, min=1e-8)

        # Common increments
        g0_increments = (
            torch.distributions.Gamma(adt, torch.ones_like(self.a))
            .rsample((batch_size, sample_size))
            .to(device)
        )

        # Add common increments to each dimension
        total_increments = g_increments + self.alpha[
            :, None, None
        ] * g0_increments.unsqueeze(0)
        return total_increments

    def simulate(self, batch_size=1):
        all_paths_increments = self.simulate_increments(batch_size)
        paths = torch.cumsum(all_paths_increments, dim=2)
        zeros = torch.zeros((self.dim, batch_size, 1), device=device, dtype=paths.dtype)
        paths = torch.cat([zeros, paths], dim=2)
        return paths


class WVAGProcess:
    def __init__(self, a, alpha, mu, theta, sigma_sqrt):
        self.ag_subordinator = AGProcess(a, alpha)
        self.dim = self.ag_subordinator.dim

        # Do not rewrap with torch.tensor() or .clone().detach()
        self.mu = mu.reshape(self.dim, 1, 1).to(dtype=DT, device=device)
        self.theta = theta.reshape(self.dim, 1, 1).to(dtype=DT, device=device)
        self.sigma_sqrt = sigma_sqrt.to(dtype=DT, device=device)

    def simulate(self, batch_size=1):
        ag_increments = self.ag_subordinator.simulate_increments(batch_size)

        # Standard Brownian increments
        wg_increments = torch.sqrt(torch.abs(ag_increments)) * torch.randn(
            self.dim, batch_size, sample_size, device=device, dtype=DT
        )

        # Generating the AG path
        ag_paths = torch.cumsum(ag_increments, dim=2)
        ag_paths = torch.cat(
            [
                torch.zeros(
                    (self.dim, batch_size, 1), device=device, dtype=ag_paths.dtype
                ),
                ag_paths,
            ],
            dim=2,
        )

        # Generating the Brownian Motion path
        wg_paths = torch.cumsum(wg_increments, dim=2)
        wg_paths = torch.cat(
            [
                torch.zeros(
                    (self.dim, batch_size, 1), device=device, dtype=wg_paths.dtype
                ),
                wg_paths,
            ],
            dim=2,
        )

        # Apply loading matrix
        wg_paths = torch.einsum("ij,jbt->ibt", self.sigma_sqrt, wg_paths)

        # Combining everything to obtain the log price
        xt_paths = self.mu * t_values.view(1, 1, -1) + self.theta * ag_paths + wg_paths

        return xt_paths

    def simulate_terminal(self, batch_size=1):
        # Efficient endpoint-only simulation
        ag_increments = self.ag_subordinator.simulate_increments(batch_size)
        ag_terminal = ag_increments.sum(dim=2)

        wg_terminal = torch.sqrt(torch.abs(ag_terminal)) * torch.randn(
            self.dim, batch_size, device=device, dtype=DT
        )
        wg_terminal = torch.einsum("ij,jb->ib", self.sigma_sqrt, wg_terminal)

        xt_terminal = self.mu[:, :, 0] + self.theta[:, :, 0] * ag_terminal + wg_terminal

        return xt_terminal  # [dim, batch_size]

    def char_function(self, frequency_argument, time_to_maturity):

        # Drift
        mu_term = 1j * frequency_argument * (self.mu * time_to_maturity)

        cumulant_argument = 1j * self.theta * frequency_argument - 0.5 * (
            frequency_argument**2
        ) * (self.sigma_sqrt**2)

        a_param = self.ag_subordinator.a
        alpha_param = self.ag_subordinator.alpha

        common_gamma_shape = a_param * time_to_maturity

        idio_gamma_shape = (
            (1 - a_param * alpha_param) / alpha_param
        ) * time_to_maturity
        # idio_gamma_scale = 1.0 / alpha_param
        idio_gamma_scale = alpha_param

        # MGF of the idiosyncratic part: (1 − (1/α)·s)^(−βT)
        phi_idio = (1 - idio_gamma_scale * cumulant_argument) ** (-idio_gamma_shape)

        # MGF of the common part: (1 − α·s)^(−shape_comm)
        phi_common = (1 - alpha_param * cumulant_argument) ** (-common_gamma_shape)

        # combine drift, idiosyncratic, and common pieces
        return torch.exp(mu_term) * phi_idio * phi_common
