"""Loading the two raw files and making the same train/test split as the original notebook."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "churn"
ID_COL = "Customer_ID"


def load_data(data_dir) -> pd.DataFrame:
    """Read Client.csv and Record.csv from data_dir and join them on Customer_ID (one row per customer)."""
    data_dir = Path(data_dir)
    client = pd.read_csv(data_dir / "Client.csv")
    record = pd.read_csv(data_dir / "Record.csv")
    return pd.merge(client, record, on=ID_COL, how="inner")


def split_train_test(df: pd.DataFrame, test_size=0.2, seed=42):
    """Stratified 80/20 split. Same settings as the original notebook, so the test customers are the same."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
