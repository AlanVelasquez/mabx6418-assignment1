"""Load the raw Amazon reviews and provide sampling utilities."""
import json
import os
import pandas as pd

from . import config


def load_reviews() -> pd.DataFrame:
    """Load the JSONL file into a typed dataframe (cached parquet on 2nd use)."""
    if os.path.exists(config.RAW_JSONL):
        records = []
        with open(config.RAW_JSONL, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        df = pd.DataFrame(records)
    else:
        raise FileNotFoundError(f"Missing {config.RAW_JSONL}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["rating"] = df["rating"].astype(float).round(0).astype(int)
    df["helpful_vote"] = df["helpful_vote"].fillna(0).astype(int)
    df["verified_purchase"] = df["verified_purchase"].astype(bool)
    df["year"] = df["timestamp"].dt.year
    df["text"] = df["text"].astype(str)
    df["title"] = df["title"].astype(str)
    return df


def balanced_sample(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """Draw an equal ~per_class-sized sample from each rating class in the WHOLE file.

    Uses a fixed random seed so the same rows come up every run.
    """
    out = []
    for cls, fn in [("NEGATIVE", config.rating_to_three),
                    ("NEUTRAL", config.rating_to_three),
                    ("POSITIVE", config.rating_to_three)]:
        mask = df["rating"].apply(lambda r: fn(r) == cls)
        sub = df[mask]
        n = min(per_class, len(sub))
        picked = sub.sample(n=n, random_state=seed)
        picked = picked.copy()
        picked["_class"] = cls
        out.append(picked)
    return pd.concat(out, ignore_index=True)
