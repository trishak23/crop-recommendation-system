"""Crop recommendation + rule-based soil health advice.

Usage:
    python -m src.recommend --N 90 --P 42 --K 43 --temperature 25 --humidity 80 --ph 6.5 --rainfall 200
"""
from __future__ import annotations

import argparse

import joblib
import pandas as pd

from .config import CROP_PROFILES, FEATURES, MODEL_PATH


def soil_advice(crop: str, N: float, P: float, K: float, ph: float, rainfall: float, temperature: float) -> list[str]:
    """Compare the field's conditions with the crop's ideal profile and suggest actions."""
    prof = CROP_PROFILES[crop]
    tips = []
    for name, value, label, fix in [
        ("N", N, "Nitrogen", "apply urea / FYM / green manure"),
        ("P", P, "Phosphorus", "apply single super phosphate (SSP) or DAP"),
        ("K", K, "Potassium", "apply muriate of potash (MOP) or compost"),
    ]:
        ideal = prof[name]
        if value < 0.75 * ideal:
            tips.append(f"{label} is low ({value:.0f} vs ~{ideal} ideal): {fix}.")
        elif value > 1.35 * ideal:
            tips.append(f"{label} is high ({value:.0f} vs ~{ideal} ideal): skip {label.lower()} fertiliser this cycle.")

    if ph < prof["ph"] - 0.7:
        tips.append(f"Soil is acidic (pH {ph:.1f}): apply agricultural lime to raise pH towards {prof['ph']}.")
    elif ph > prof["ph"] + 0.7:
        tips.append(f"Soil is alkaline (pH {ph:.1f}): add gypsum, sulphur or organic matter to lower pH towards {prof['ph']}.")

    if rainfall < 0.7 * prof["rainfall"]:
        tips.append(f"Rainfall ({rainfall:.0f} mm) is below the ~{prof['rainfall']} mm {crop} needs: plan supplementary irrigation (drip if possible).")
    elif rainfall > 1.5 * prof["rainfall"]:
        tips.append(f"Rainfall ({rainfall:.0f} mm) is high for {crop}: ensure good drainage to avoid waterlogging.")

    if abs(temperature - prof["temperature"]) > 6:
        tips.append(f"Temperature ({temperature:.0f}°C) is far from the ~{prof['temperature']}°C {crop} prefers: adjust sowing date to the {prof['season']} window.")
    return tips or ["Conditions are well matched to this crop. Maintain standard good practice."]


class CropRecommender:
    def __init__(self, model=None):
        self.model = model if model is not None else joblib.load(MODEL_PATH)

    def top_k(self, N, P, K, temperature, humidity, ph, rainfall, k: int = 3) -> pd.DataFrame:
        row = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=FEATURES)
        proba = pd.Series(self.model.predict_proba(row)[0], index=self.model.classes_)
        top = proba.sort_values(ascending=False).head(k)
        return pd.DataFrame({
            "crop": top.index,
            "confidence": top.values.round(3),
            "season": [CROP_PROFILES[c]["season"] for c in top.index],
        })

    def recommend(self, N, P, K, temperature, humidity, ph, rainfall, k: int = 3) -> dict:
        top = self.top_k(N, P, K, temperature, humidity, ph, rainfall, k)
        best = top.iloc[0]["crop"]
        return {
            "top_crops": top,
            "advice": soil_advice(best, N, P, K, ph, rainfall, temperature),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop recommendation")
    for feat in FEATURES:
        parser.add_argument(f"--{feat}", type=float, required=True)
    args = vars(parser.parse_args())
    result = CropRecommender().recommend(**args)
    print("\nRecommended crops:\n")
    print(result["top_crops"].to_string(index=False))
    print("\nSoil & climate advice for the top crop:")
    for tip in result["advice"]:
        print(f"  - {tip}")


if __name__ == "__main__":
    main()
