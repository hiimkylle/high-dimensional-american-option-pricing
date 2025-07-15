import numpy as np
import torch


def moment_errors(data, cal):
    emp_mean, emp_var, emp_skew, emp_kurt = _empirical_moments(data)
    mdl_mean, mdl_var, mdl_skew, mdl_kurt = cal.compute_calibrated_moments()

    mse = lambda x, y: np.mean((x - y) ** 2)
    return (
        mse(emp_mean, mdl_mean),
        mse(emp_var, mdl_var),
        mse(emp_skew, mdl_skew),
        mse(emp_kurt, mdl_kurt),
    )


def corr_error(data, cal):
    emp_corr = np.corrcoef(data.cpu().numpy(), rowvar=False)
    mdl_corr = np.corrcoef(
        cal._sim_cache[next(iter(cal._sim_cache))].cpu().numpy(), rowvar=False
    )
    return np.linalg.norm(emp_corr - mdl_corr, "fro")


def _empirical_moments(data_t):
    x = data_t.cpu().numpy()
    mean = x.mean(0)
    centered = x - mean
    var = centered.var(0, ddof=1)
    skew = (centered**3).mean(0) / (var**1.5 + 1e-12)
    kurt = (centered**4).mean(0) / (var**2 + 1e-12)
    return mean, var, skew, kurt
