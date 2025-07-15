from config import *
from dme_marginal import MarginalCalibrator
from scipy import optimize
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm
import numpy as np
import wvag as sim
import torch


def batched_simulate_terminal(wv_process, n_mc, batch):
    all_samples = []

    for start in range(0, n_mc, batch):
        cur_batch = min(batch, n_mc - start)
        samples = wv_process.simulate_terminal(cur_batch)
        all_samples.append(samples)

    return torch.cat(all_samples, dim=1)


class DMECalibrator:
    def __init__(self, data, t_horizon=1.0):
        self.data = data
        self.probs1 = probs1
        self.probs2 = probs2
        self.t_horizon = t_horizon
        self.d = data.shape[1]

        p2_t = torch.tensor(probs2, dtype=DT, device=device)
        self.k2 = torch.quantile(data, p2_t, dim=0).cpu().numpy()

    def fit_marginals(self, **kwargs):
        marginal_cal = MarginalCalibrator(
            self.data, self.probs1, t_horizon=self.t_horizon
        )
        results = marginal_cal.fit(**kwargs)

        self.alpha, self.theta, self.sigma, self.mu = results
        self.alpha = torch.maximum(
            self.alpha,
            torch.tensor(1e-6, device=self.alpha.device, dtype=self.alpha.dtype),
        )

        return results

    def _empirical_joint_digitals(self, loess_frac=0.3):
        d = self.d
        idx_i, idx_j = torch.tril_indices(d, d, offset=-1)
        data = self.data

        k2_tensor = torch.tensor(self.k2, dtype=DT, device=device)
        ki = k2_tensor[:, idx_i]
        kj = k2_tensor[:, idx_j]

        xi = data[:, idx_i]
        xj = data[:, idx_j]

        xi_b = xi.unsqueeze(1)
        xj_b = xj.unsqueeze(1)

        ki_b = ki.unsqueeze(0)
        kj_b = kj.unsqueeze(0)

        mask = (xi_b <= ki_b) & (xj_b <= kj_b)
        vals = mask.float().mean(dim=0)
        emps = vals.permute(1, 0).contiguous()

        emps_np = emps.cpu().numpy()
        n_points = emps_np.shape[1]
        smoothed = np.zeros_like(emps_np)
        x_quant = np.arange(n_points)

        for i in range(emps_np.shape[0]):
            fit = lowess(emps_np[i], x_quant, frac=loess_frac, return_sorted=False)
            smoothed[i] = fit

        return torch.tensor(smoothed, dtype=emps.dtype, device=emps.device)

    def _model_joint_digitals(self, L, a_val, n_mc=n_mc, batch=batch):
        # Setting up cache
        if not hasattr(self, "_sim_cache"):
            self._sim_cache = {}
        L_hash = tuple(np.round(L.detach().cpu().numpy().flatten(), 8))
        cache_key = (L_hash, float(a_val))

        if cache_key in self._sim_cache:
            model_data = self._sim_cache[cache_key]
        else:
            sig_sqrt = torch.diag(self.sigma) @ L

            a_max = (1.0 - 1e-6) / self.alpha.max().item()
            a_val = float(np.clip(a_val, 1e-8, a_max))

            alpha_clamped = torch.clamp(
                self.alpha, min=1e-8, max=(1.0 - 1e-6) / a_val
            ).to(dtype=DT, device=device)

            mu_safe = torch.clamp(self.mu, min=-1e8, max=1e8).to(
                dtype=DT, device=device
            )
            theta_safe = torch.clamp(self.theta, min=-1e8, max=1e8).to(
                dtype=DT, device=device
            )

            sig_sqrt_safe = sig_sqrt.to(dtype=DT, device=device)

            wv = sim.WVAGProcess(
                a_val, alpha_clamped, mu_safe, theta_safe, sig_sqrt_safe
            )
            all_incs = []

            for start in range(0, n_mc, batch):
                cur = min(batch, n_mc - start)
                inc = wv.simulate_terminal(cur).T
                all_incs.append(inc)

            model_data = torch.cat(all_incs, 0)
            self._sim_cache[cache_key] = model_data

        d = self.d
        n_q = len(self.probs2)
        idx_i, idx_j = torch.tril_indices(d, d, offset=-1)

        k2_tensor = torch.tensor(self.k2, dtype=DT, device=device)
        ki = k2_tensor[:, idx_i]
        kj = k2_tensor[:, idx_j]

        xi = model_data[:, idx_i]
        xj = model_data[:, idx_j]

        xi_b = xi.unsqueeze(1)
        xj_b = xj.unsqueeze(1)

        ki_b = ki.unsqueeze(0)
        kj_b = kj.unsqueeze(0)

        mask = (xi_b <= ki_b) & (xj_b <= kj_b)
        vals = mask.float().mean(dim=0)
        mods = vals.permute(1, 0).contiguous()

        return mods

    def fit_dependence(self):

        R0 = np.corrcoef(self.data.cpu().numpy(), rowvar=False)
        L0 = np.linalg.cholesky(R0 + 1e-6 * np.eye(self.d))

        idx = torch.tril_indices(self.d, self.d, offset=-1)
        v0 = L0[idx[0], idx[1]].copy()
        log_a0 = np.log(0.5)
        v0_1d = np.atleast_1d(v0)
        v0_full = np.concatenate([v0_1d, [log_a0]])

        self._dep_pbar = tqdm(total=1000, desc="Calibrating dependence")
        self._dep_calls = 0

        def loss(v_np):
            self._dep_calls += 1

            if self._dep_calls % 3 == 0:
                self._dep_pbar.update(1)

            v = torch.tensor(v_np[:-1], dtype=DT, device=device)
            log_a = v_np[-1]
            a = np.exp(log_a)
            a = np.clip(a, 1e-8, 0.99 / self.alpha.max().item())

            if np.any(np.isnan(v_np)) or np.isnan(a) or a <= 0:
                print("NaN or invalid parameter in input to loss!", v_np, "a:", a)
                return 1e10

            if joint_box_a_max is not None:
                penalty_a_box = 1e8 * np.maximum(np.log(a) - np.log(joint_box_a_max), 0)
            else:
                penalty_a_box = 0.0

            L = torch.eye(self.d, dtype=DT, device=device)
            L[idx[0], idx[1]] = v
            L = L / L.norm(dim=1, keepdim=True)

            emp = self._empirical_joint_digitals(loess_frac=loess_frac)
            mod = self._model_joint_digitals(L, a)

            Sigma = torch.diag(self.sigma) @ (L @ L.T) @ torch.diag(self.sigma)
            logdet = torch.logdet(Sigma)

            penalty = (
                reg_sigma_large * (self.sigma**2).sum()
                + reg_sigma_small * (1.0 / self.sigma**2).sum()
                + reg_l2 * (self.sigma**2).sum()
                + reg_a * (a**2)
                + 1e8 * np.maximum(-log_a, 0)  # ensure log_a > 0
                + 1e6 * max(0.0, a * self.alpha.max().item() - 0.95)
                + 0.1 * (a**4)
                + penalty_a_box
                + reg_logdet * (-logdet)
            )

            lower_a, upper_a = log_a_bounds

            penalty += soft_a_penalty_weight * (
                np.maximum(lower_a - log_a, 0.0) ** 2
                + np.maximum(log_a - upper_a, 0.0) ** 2
            )

            if torch.isnan(L).any():
                return 1e10

            lossval = ((emp - mod) ** 2).sum().item() + penalty.item()

            # Joint Penalties
            L_hash = tuple(np.round(L.detach().cpu().numpy().flatten(), 8))
            model_sample = self._sim_cache[(L_hash, float(a))].cpu().numpy()

            emp_data = self.data.cpu().numpy()
            emp_cov = np.cov(emp_data, rowvar=False)
            model_cov = np.cov(model_sample, rowvar=False)
            penalty_cov = reg_cov * np.sum((emp_cov - model_cov) ** 2)

            emp_corr = np.corrcoef(emp_data, rowvar=False)
            model_corr = np.corrcoef(model_sample, rowvar=False)
            offdiag_mask = ~np.eye(model_cov.shape[0], dtype=bool)
            penalty_offdiag = reg_offdiag * np.sum(model_cov[offdiag_mask] ** 2)
            penalty_corr = reg_corr * np.sum((emp_corr - model_corr) ** 2)
            lossval += penalty_offdiag + penalty_corr + penalty_cov

            if np.isnan(lossval):
                return 1e10

            return lossval

        log_a_bounds = (np.log(1e-8), np.log(0.99 / self.alpha.max().item()))
        v_bounds = [(-10, 10)] * len(v0_1d) + [log_a_bounds]
        res = optimize.minimize(
            loss,
            v0_full,
            method="L-BFGS-B",
            bounds=v_bounds,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        self._dep_pbar.close()

        v_opt = torch.tensor(res.x[:-1], dtype=DT, device=device)
        log_a_opt = res.x[-1]
        self.a = float(np.exp(log_a_opt))

        L = torch.eye(self.d, dtype=DT, device=device)
        L[idx[0], idx[1]] = v_opt
        L = L / L.norm(dim=1, keepdim=True)
        self.L = L

        return L

    def fit(self):

        self.fit_marginals()

        self.fit_dependence()

    @property
    def Sigma(self):
        # Note that Sigma = diag(sigma) @ L @ L.T @ diag(sigma)
        return torch.diag(self.sigma) @ (self.L @ self.L.T) @ torch.diag(self.sigma)

    # For testing: return marginal momenta
    def compute_calibrated_moments(self):
        theta = self.theta.cpu().numpy()
        sigma = self.sigma.cpu().numpy()
        alpha = self.alpha.cpu().numpy()
        mu = self.mu.cpu().numpy()

        nu = 1.0 / alpha
        mean = mu + nu * theta
        var = sigma**2 + nu * theta**2
        skew = (theta * nu * (3 * sigma**2 + 2 * nu * theta**2)) / (var**1.5)
        kurt = 3 + (
            3 * nu * (sigma**4 + 2 * sigma**2 * nu * theta**2 + nu**2 * theta**4)
        ) / (var**2)

        return mean, var, skew, kurt
