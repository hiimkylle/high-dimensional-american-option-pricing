import torch
import pandas as pd
from bootstrap import bootstrap_calibration
from config import *


def main():
    data_logret_dataframe = pd.read_csv(
        "data_logret_indices.csv", index_col="Date", parse_dates=True
    )

    input_data_tensor = torch.tensor(
        data_logret_dataframe.to_numpy(), dtype=DT, device=device
    )

    data_tensor = input_data_tensor  # (n, d) torch.double on correct device
    df_boot = bootstrap_calibration(data_tensor, n_boot=5)

    print("\n===== Bootstrap summary =====")
    print(df_boot.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T)

    # maybe: stop if median Frobenius error > threshold
    med_frob = df_boot["frob_corr"].median()
    print(f"Median corr-Frob error : {med_frob: .4e}")


if __name__ == "__main__":
    main()
