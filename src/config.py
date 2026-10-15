"""Shared constants: paths, feature names and agronomic crop profiles."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "crop_data.csv"
MODEL_PATH = ROOT / "models" / "crop_model.joblib"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "crop"

FEATURE_LABELS = {
    "N": "Nitrogen (kg/ha)",
    "P": "Phosphorus (kg/ha)",
    "K": "Potassium (kg/ha)",
    "temperature": "Temperature (°C)",
    "humidity": "Relative humidity (%)",
    "ph": "Soil pH",
    "rainfall": "Rainfall (mm)",
}

# Approximate ideal conditions per crop (typical of Indian growing conditions).
# These drive both the synthetic data and the soil-health advice. Replace with
# agronomic values from your state agriculture university for production use.
CROP_PROFILES = {
    #               N    P    K   temp  hum   pH   rain  season
    "Rice":        dict(N=80,  P=48, K=40, temperature=24, humidity=82, ph=6.4, rainfall=236, season="Kharif"),
    "Wheat":       dict(N=100, P=45, K=35, temperature=18, humidity=60, ph=6.8, rainfall=75,  season="Rabi"),
    "Maize":       dict(N=78,  P=48, K=20, temperature=22, humidity=65, ph=6.2, rainfall=85,  season="Kharif"),
    "Cotton":      dict(N=118, P=46, K=20, temperature=24, humidity=80, ph=6.9, rainfall=80,  season="Kharif"),
    "Sugarcane":   dict(N=120, P=50, K=50, temperature=27, humidity=75, ph=7.0, rainfall=150, season="Annual"),
    "Soybean":     dict(N=25,  P=60, K=40, temperature=25, humidity=70, ph=6.5, rainfall=110, season="Kharif"),
    "Jowar":       dict(N=80,  P=40, K=30, temperature=28, humidity=55, ph=7.0, rainfall=60,  season="Kharif/Rabi"),
    "Bajra":       dict(N=60,  P=30, K=30, temperature=30, humidity=45, ph=7.2, rainfall=45,  season="Kharif"),
    "Chickpea":    dict(N=40,  P=68, K=80, temperature=19, humidity=17, ph=7.3, rainfall=80,  season="Rabi"),
    "Groundnut":   dict(N=25,  P=55, K=40, temperature=28, humidity=65, ph=6.3, rainfall=100, season="Kharif"),
    "Banana":      dict(N=100, P=80, K=50, temperature=27, humidity=80, ph=6.0, rainfall=105, season="Perennial"),
    "Mango":       dict(N=20,  P=25, K=30, temperature=31, humidity=50, ph=5.8, rainfall=95,  season="Perennial"),
}
