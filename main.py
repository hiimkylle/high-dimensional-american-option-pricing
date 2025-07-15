from dme_main import batched_simulate_terminal
from scipy.stats import skew, kurtosis
from wvag import WVAGProcess
from visualisation import joint_structure_plots
from data_scraping import obtain_data
import pandas as pd
import torch
from config import *
from dme_main import DMECalibrator


def cov_to_corr(Sigma):
    D = np.sqrt(np.diag(Sigma))
    return Sigma / np.outer(D, D)


def print_summary_marginals(df_vals, dme):
    emp_mean = df_vals.mean(axis=0)
    emp_var = df_vals.var(axis=0, ddof=1)  #
    emp_skew = skew(df_vals, axis=0, bias=False)
    emp_kurt = kurtosis(
        df_vals, axis=0, bias=False, fisher=False
    )  # Pearson/Regular kurtosis

    cal_mean, cal_var, cal_skew, cal_kurt = dme.compute_calibrated_moments()

    print("\n--- Marginal Moment Comparison ---")
    for i in range(df_vals.shape[1]):
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


def print_summary_joint(df_vals, dme):
    Sigma_emp = torch.cov(
        torch.tensor(df_vals, dtype=DT, device=device).T, correction=1
    )

    print("True correlation matrix:\n", cov_to_corr(Sigma_emp.cpu().numpy()))
    print("Fitted correlation matrix:\n", cov_to_corr(dme.Sigma.cpu().numpy()))

    sigma_sqrt = torch.linalg.cholesky(dme.Sigma)
    wvag_fitted = WVAGProcess(dme.a, dme.alpha, dme.mu, dme.theta, sigma_sqrt)
    data_model = (
        batched_simulate_terminal(wvag_fitted, n_mc=n_mc, batch=batch).T.cpu().numpy()
    )
    joint_structure_plots(df_vals, data_model, sample=df_vals.shape[0])


def main():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    # Obtain data and convert to tensor
    obtain_data()

    data_logret_dataframe = pd.read_csv(
        "data_logret_indices.csv", index_col="Date", parse_dates=True
    )

    input_data_tensor = torch.tensor(
        data_logret_dataframe.to_numpy(), dtype=DT, device=device
    )

    # Device, data and integer representation info
    print("Tensor shape (n_datapoints, n_indices):", input_data_tensor.shape)
    print("Device:", input_data_tensor.device)
    print("Integer representation:", DT)

    # Perform DME Calibration
    dme = DMECalibrator(input_data_tensor)
    print("Callibrating...")
    dme.fit()

    # Statistics Report
    print("Printing summary...")

    print("\nCalibrated parameters:")
    print("a=", float(dme.a))
    print("alpha=", dme.alpha.cpu().numpy())
    print("theta=", dme.theta.cpu().numpy())
    print("mu=", dme.mu.cpu().numpy())
    print("sigma=", dme.sigma.cpu().numpy())
    print("Sigma=\n", dme.Sigma.cpu().numpy())

    print_summary_marginals(data_logret_dataframe.to_numpy(), dme)
    print_summary_joint(data_logret_dataframe.to_numpy(), dme)


if __name__ == "__main__":
    main()
