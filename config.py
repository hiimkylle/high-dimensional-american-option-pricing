from pathlib import Path
import numpy as np
import torch

# Use GPU if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DT = torch.float64

# Simulation parameters
t_max = 1.0
sample_size = 10000
delta_t = t_max / sample_size
t_values = torch.linspace(0, t_max, sample_size + 1, device=device).double()

d_test_simulator = 5

# Calibrator parameters
n_mc = 10000
batch = 2000
n_quadrature = 32

n_data = 2000  # Dictates amount of sample data generated
d_test = 5  # Dictates the dimension of test data

probs1 = np.linspace(0.05, 0.95, 10)
probs2 = np.linspace(0.05, 0.95, 10)

# DME Regularisation parameters: Balance bias and variance

# Many other regularisation parameters
reg_sigma_large = 0.0
reg_sigma_small = 0.0
reg_mu = 1.0
reg_theta = 1.0
reg_alpha = 0.0
reg_l2 = 1e-5
reg_a = 0.0
reg_logdet = 1e-5

# Marginal Penalties
reg_gmm_var = 9000.0
reg_gmm_mean = 2500.0
skew_penalty = 50.0
kurt_penalty = 5.0

# Soft quadratic penalities to lightly enforce bounds
log_alpha_bounds = (np.log(0.01), np.log(0.3))
log_sigma_bounds = (np.log(0.006), np.log(0.06))
log_a_bounds = (np.log(0.005), np.log(1.0))

soft_a_penalty_weight = 100.0
soft_alpha_penalty_weight = 250.0
soft_sigma_penalty_weight = 250.0

# Regularisation for joint dependence
reg_corr = 1000.0
reg_offdiag = 0.5
reg_cov = 800.0

# Other regularisation parameters
joint_box_a_max = 0.6
scale_penalty = 2000.0
loess_frac = 0.15

# Neural Network Parameters


# Data scrapper settings

# Downloading data
SYMBOLS = {
    "^GSPC": "S&P 500",
    "NDX": "Tesla, Inc.",
    "XOM": "Exxon Mobil Corporation",
    "JPM": "JPMorgan Chase & Co.",
    "WMT": "Walmart Inc.",
}
START_DATE = "2015-01-02"
END_DATE = "2025-01-02"  # inclusive end date
OUTFILE = Path("data/raw/indices_close.csv")

FILE_CLOSES = Path("data_raw_indices.csv")
FILE_LOGRET = Path("data_logret_indices.csv")
