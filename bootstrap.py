import numpy as np
import torch
import pandas as pd
from tqdm.auto import tqdm

from dme_main import DMECalibrator  # uses your current regularisation cfg
from utils import moment_errors, corr_error  # helper functions defined below


def bootstrap_calibration(
    data_tensor: torch.Tensor,
    n_boot: int = 50,
    seed: int | None = 42,
    silent: bool = False,
):
    if seed is not None:
        np.random.seed(seed)

    n, d = data_tensor.shape
    rows = np.arange(n)

    records = []
    iterator = range(n_boot)
    if not silent:
        iterator = tqdm(iterator, desc="Bootstrapping", leave=False)

    for b in iterator:
        # 1. sample rows with replacement
        idx = np.random.choice(rows, size=n, replace=True)
        sample = data_tensor[idx]

        # 2. fresh calibrator
        cal = DMECalibrator(sample)
        cal.fit()  # <- heavy optimisation step

        # 3. diagnostics
        mse_mu, mse_var, mse_skew, mse_kurt = moment_errors(sample, cal)
        corr_frob = corr_error(sample, cal)

        rec = {
            "boot": b,
            "mse_mean": mse_mu,
            "mse_var": mse_var,
            "mse_skew": mse_skew,
            "mse_kurt": mse_kurt,
            "frob_corr": corr_frob,
            "a": cal.a,
        }
        # flatten vectors for quick eyeballing
        rec.update({f"alpha_{i}": v for i, v in enumerate(cal.alpha.cpu().numpy())})
        rec.update({f"sigma_{i}": v for i, v in enumerate(cal.sigma.cpu().numpy())})
        records.append(rec)

    return pd.DataFrame.from_records(records)
