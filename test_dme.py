from config import d_test
from dme_main import (
    DMECalibrator,
    batch,
    batched_simulate_terminal,
    device,
    n_data,
    n_mc,
    torch,
)
from visualisation import joint_structure_plots
from scipy.stats import skew, kurtosis
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wvag as sim
import pandas as pd


def cov_to_corr(Sigma):
    D = np.sqrt(np.diag(Sigma))
    return Sigma / np.outer(D, D)


if __name__ == "__main__":
    import parameter_generator as pg

    print("Using device:", device)
    if device.type == "cuda":
        print("CUDA device name:", torch.cuda.get_device_name(0))
    else:
        print("CUDA not available, using CPU.")

    # Setting true parameters
    a0, a1, mu0, th0, sig0 = pg.draw_wvag_params(d=d_test, device=device)
    Sigma_true = sig0 @ sig0.T
    sigma_true = torch.sqrt(torch.diag(Sigma_true))

    # Simulating data
    wv = sim.WVAGProcess(a0, a1, mu0, th0, sig0)
    X = batched_simulate_terminal(wv, n_mc=n_mc, batch=batch).T

    # Initialising DME
    dme = DMECalibrator(X)

    # Running DME calibration
    dme.fit()

    # Printing statistics
    # print('True parameters:')
    # print('a=', float(a0))
    # print('alpha=', a1.cpu().numpy())
    # print('theta=', th0.cpu().numpy())
    # print('mu=', mu0.cpu().numpy())
    # print('sigma=', sigma_true.cpu().numpy())
    # print('Sigma_true=\n', Sigma_true.cpu().numpy())

    # print('\nCalibrated parameters:')
    # print('a=', float(dme.a))
    # print('alpha=', dme.alpha.cpu().numpy())
    # print('theta=', dme.theta.cpu().numpy())
    # print('mu=', dme.mu.cpu().numpy())
    # print('sigma=', dme.sigma.cpu().numpy())
    # print('Sigma=\n', dme.Sigma.cpu().numpy())

    print("True correlation matrix:\n", cov_to_corr(Sigma_true.cpu().numpy()))
    print("Fitted correlation matrix:\n", cov_to_corr(dme.Sigma.cpu().numpy()))

    # Compute empirical and calibrated mean/variance
    X_np = X.cpu().numpy()
    emp_mean = np.mean(X_np, axis=0)
    emp_var = np.var(X_np, axis=0, ddof=1)
    emp_skew = skew(X_np, axis=0, bias=False)
    emp_kurt = kurtosis(X_np, axis=0, bias=False, fisher=False)

    cal_mean, cal_var, cal_skew, cal_kurt = dme.compute_calibrated_moments()

    print("\n--- Marginal Moment Comparison ---")
    for i in range(X.shape[1]):
        print(f"Dimension {i+1}:")
        print(
            f"  Mean     → Empirical = {emp_mean[i]: .5f},  Calibrated = {cal_mean[i]: .5f}"
        )
        print(
            f"  Variance → Empirical = {emp_var[i]: .5f},  Calibrated = {cal_var[i]: .5f}"
        )
        print(
            f"  Skewness → Empirical = {emp_skew[i]: .5f},  Calibrated = {cal_skew[i]: .5f}"
        )
        print(
            f"  Kurtosis → Empirical = {emp_kurt[i]: .5f},  Calibrated = {cal_kurt[i]: .5f}"
        )

    sigma_sqrt = torch.linalg.cholesky(dme.Sigma)
    wvag_fitted = sim.WVAGProcess(dme.a, dme.alpha, dme.mu, dme.theta, sigma_sqrt)
    X_model = (
        batched_simulate_terminal(wvag_fitted, n_mc=n_mc, batch=batch).T.cpu().numpy()
    )

    # Visual Plots
    # plot_nd_marginals(X_np, X_model)
    joint_structure_plots(X_np, X_model)
