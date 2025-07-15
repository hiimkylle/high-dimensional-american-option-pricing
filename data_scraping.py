from config import *
import yfinance as yf
import pandas as pd
import numpy as np


def download_closes(symbols: dict, start: str, end: str) -> pd.DataFrame:
    """Download CLOSE prices for all symbols in one call."""
    tickers = list(symbols.keys())
    df_all = yf.download(
        tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )

    # Yahoo’s multi-ticker output: outer column level = ticker, inner = field
    # Extract 'Close' layer and rename columns.
    if isinstance(df_all.columns, pd.MultiIndex):
        df_close = df_all.xs("Close", level=1, axis=1)
        df_close = df_close.rename(columns=symbols)
    else:
        # Fallback (single ticker case)
        df_close = df_all.rename(columns={"Close": list(symbols.values())[0]})

    # Warn if any column is completely NA (download failure)
    bad = [c for c in df_close.columns if df_close[c].isna().all()]
    if bad:
        print("Warning: no data for ->", bad)

    # Drop rows with ANY NA to keep series aligned
    df_close = df_close.dropna(how="any")

    return df_close


def obtain_data():
    # 1. Download daily closes
    df_close = download_closes(SYMBOLS, START_DATE, END_DATE).dropna(how="any")
    df_close.to_csv(FILE_CLOSES, index_label="Date")
    print(f" RAW closes  saved to {FILE_CLOSES.resolve()}  {df_close.shape}")

    # 2. Compute log-returns
    df_logret = np.log(df_close / df_close.shift(1)).dropna(how="any")
    df_logret.to_csv(FILE_LOGRET, index_label="Date")
    print(f" Log-returns saved to {FILE_LOGRET.resolve()}  {df_logret.shape}")
