"""Generate a synthetic crop-suitability dataset for Indian conditions.

Each crop's samples are drawn around its ideal soil/climate profile (src/config.py) with
realistic spread, so neighbouring crops overlap (e.g. Jowar vs Bajra, Rice vs Banana).

To use the Kaggle "Crop Recommendation Dataset" instead, save it as data/crop_data.csv with
columns N, P, K, temperature, humidity, ph, rainfall, crop (lowercase labels work too).

Usage:
    python -m src.generate_data --per-crop 250 --seed 42
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import CROP_PROFILES, DATA_PATH, FEATURES, TARGET

SPREAD = {"N": 14, "P": 9, "K": 9, "temperature": 3.0, "humidity": 9, "ph": 0.4}
BOUNDS = {
    "N": (0, 140), "P": (5, 145), "K": (5, 205), "temperature": (8, 44),
    "humidity": (14, 100), "ph": (3.5, 9.5), "rainfall": (20, 300),
}


def generate(per_crop: int = 250, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for crop, prof in CROP_PROFILES.items():
        data = {}
        for feat in FEATURES:
            std = max(12, 0.18 * prof[feat]) if feat == "rainfall" else SPREAD[feat]
            lo, hi = BOUNDS[feat]
            data[feat] = np.clip(rng.normal(prof[feat], std, per_crop), lo, hi)
        data[TARGET] = crop
        frames.append(pd.DataFrame(data))
    df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    df[["N", "P", "K"]] = df[["N", "P", "K"]].round().astype(int)
    return df.round({"temperature": 1, "humidity": 1, "ph": 2, "rainfall": 1})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-crop", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    df = generate(args.per_crop, args.seed)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df):,} samples of {df[TARGET].nunique()} crops -> {DATA_PATH}")


if __name__ == "__main__":
    main()
