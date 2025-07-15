from config import *
import numpy as np
import torch
import tqdm


def prefit_sigma_theta(var_emp, skew_emp, nu, eps=1e-12):
    v, s = var_emp, skew_emp
    A = s * v**1.5 / nu  # cubic:  nu θ³ – 3 v θ + A = 0
    roots = np.roots([nu, 0.0, -3.0 * v, A])
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-8]

    if not real_roots:  # rare numerical fall-back
        theta = np.sign(s) * np.sqrt(max(v / nu, eps))
    else:  # choose root with correct sign
        same_sign = [r for r in real_roots if np.sign(r) == np.sign(s)]
        theta = same_sign[0] if same_sign else real_roots[0]

    sigma2 = max(v - theta**2 * nu, eps)
    return np.sqrt(sigma2), theta


# Numerically computes the VG cdf
def vg_cdf(x, a, theta, sigma, m):

    lag_t, lag_w = np.polynomial.laguerre.laggauss(n_quadrature)
    lag_t = torch.tensor(lag_t, dtype=DT, device=device)
    lag_w = torch.tensor(lag_w, dtype=DT, device=device)
    lag_w_div_t = (lag_w / lag_t)[:, None]

    psi = (1 / a) * torch.log(
        1 - 1j * theta * a * lag_t[:, None] + 0.5 * sigma**2 * a * lag_t[:, None] ** 2
    )

    expo = torch.exp(psi - 1j * lag_t[:, None] * (x - m))
    res = torch.sum(lag_w_div_t * torch.imag(expo), dim=0)

    return 0.5 - res / np.pi


class MarginalCalibrator:
    def __init__(self, data, probs, t_horizon=1.0):
        self.data = data
        self.probs = torch.tensor(probs, dtype=DT, device=device)
        self.n, self.d = data.shape
        self.t_horizon = t_horizon

        self.emp_mean = self.data.mean(dim=0)
        self.emp_var = self.data.var(dim=0, unbiased=True)

    def fit_one(self, k):
        xk = self.data[:, k]
        kv = np.quantile(xk.cpu().numpy(), self.probs.cpu().numpy())
        emp_mean = self.emp_mean[k].item()
        emp_var = self.emp_var[k].item()

        xk_centered = self.data[:, k] - self.emp_mean[k]
        emp_skew = torch.mean(xk_centered**3) / (self.emp_var[k] ** 1.5)
        emp_kurt = torch.mean(xk_centered**4) / (self.emp_var[k] ** 2)

        alpha_init = 1.0  # keep your old α guess (ν = 1)
        nu_init = 1.0 / alpha_init
        sigma0, theta0 = prefit_sigma_theta(emp_var, emp_skew.item(), nu_init)

        u0 = torch.tensor(
            [np.log(alpha_init), theta0, np.log(sigma0), self.emp_mean[k]],
            dtype=DT,
            device=device,
            requires_grad=True,
        )

        u = u0.clone().detach().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [u],
            max_iter=500,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            alpha = torch.exp(u[0])
            th = u[1]
            s = torch.exp(u[2])
            m = u[3]
            a_fixed = 1.0
            model = vg_cdf(torch.tensor(kv, device=device), a_fixed, th, s, m)
            diff = model - self.probs

            nu = 1.0 / alpha
            model_var = s**2 + nu * th**2
            model_skew = (th * nu * (3 * s**2 + 2 * nu * th**2)) / (model_var**1.5)
            model_kurt = 3 + (
                3 * nu * (s**4 + 2 * s**2 * nu * th**2 + nu**2 * th**4)
            ) / (model_var**2)

            penalty = (
                reg_sigma_large * s**2
                + reg_sigma_small / (s**2)
                + reg_mu * m**2
                + reg_theta * th**2
                + reg_alpha * alpha**2
                + reg_l2 * (alpha**2 + th**2 + s**2 + m**2)
                + reg_gmm_var * (model_var - emp_var) ** 2
                + reg_gmm_mean * (m - emp_mean) ** 2
                + scale_penalty * (alpha * s**2 - emp_var) ** 2
            )

            # Soft quadratic penalty for alpha and sigma
            lower_a, upper_a = log_alpha_bounds
            lower_s, upper_s = log_sigma_bounds

            penalty += soft_alpha_penalty_weight * (
                torch.relu(lower_a - u[0]) ** 2 + torch.relu(u[0] - upper_a) ** 2
            )

            penalty += soft_sigma_penalty_weight * (
                torch.relu(lower_s - u[2]) ** 2 + torch.relu(u[2] - upper_s) ** 2
            )

            loss = (diff**2).sum() + penalty

            # Adding skew/kurtosis penalities
            loss = (
                loss
                + skew_penalty * (model_skew - emp_skew) ** 2
                + kurt_penalty * (model_kurt - emp_kurt) ** 2
            )

            loss.backward()
            return loss

        optimizer.step(closure)

        log_alpha, th, log_sigma, m0 = [float(v) for v in u.detach()]
        mu_k = m0 / self.t_horizon
        return np.exp(log_alpha), th, np.exp(log_sigma), mu_k

    def fit(self):
        alpha, theta, sigma, mu = [], [], [], []
        for k in tqdm.tqdm(range(self.d), desc="Calibrating marginals"):
            a, t, s, mu_k = self.fit_one(k)

            alpha.append(a)
            theta.append(t)
            sigma.append(s)
            mu.append(mu_k)

        return (
            torch.tensor(alpha, device=device, dtype=DT),
            torch.tensor(theta, device=device),
            torch.tensor(sigma, device=device),
            torch.tensor(mu, device=device),
        )
