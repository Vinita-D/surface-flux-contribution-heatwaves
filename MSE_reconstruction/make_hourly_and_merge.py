import pandas as pd

def standardize_time(df: pd.DataFrame, time_col="release_time") -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df["release_time"] = df[time_col].dt.floor("H")
    return df

def hourly_mean(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    # assumes release_hour exists
    return (
        df.groupby("release_time", as_index=False)[value_cols]
        .mean(numeric_only=True)
    )