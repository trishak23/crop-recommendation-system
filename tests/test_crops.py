import pytest

from src.config import CROP_PROFILES, FEATURES
from src.generate_data import generate
from src.recommend import CropRecommender, soil_advice
from src.train import candidates


@pytest.fixture(scope="module")
def df():
    return generate(per_crop=120, seed=5)


@pytest.fixture(scope="module")
def recommender(df):
    model = candidates()["Gaussian Naive Bayes"].fit(df[FEATURES], df["crop"])
    return CropRecommender(model=model)


def test_generated_data(df):
    assert df["crop"].nunique() == len(CROP_PROFILES)
    assert len(df) == 120 * len(CROP_PROFILES)
    assert df["ph"].between(3.5, 9.5).all()
    assert df.isna().sum().sum() == 0


def test_ideal_profile_predicts_own_crop(recommender):
    clear_cases = ["Rice", "Chickpea", "Banana", "Mango", "Sugarcane"]
    for crop in clear_cases:
        p = CROP_PROFILES[crop]
        top = recommender.top_k(p["N"], p["P"], p["K"], p["temperature"], p["humidity"], p["ph"], p["rainfall"], k=1)
        assert top.iloc[0]["crop"] == crop


def test_top_k_probabilities_valid(recommender):
    top = recommender.top_k(80, 45, 40, 25, 70, 6.5, 100, k=3)
    assert len(top) == 3
    assert top["confidence"].is_monotonic_decreasing
    assert top["confidence"].sum() <= 1.0001


def test_soil_advice_flags_problems():
    tips = soil_advice("Rice", N=20, P=48, K=40, ph=4.8, rainfall=90, temperature=24)
    text = " ".join(tips).lower()
    assert "nitrogen" in text and "acidic" in text and "irrigation" in text


def test_soil_advice_all_good():
    p = CROP_PROFILES["Wheat"]
    tips = soil_advice("Wheat", p["N"], p["P"], p["K"], p["ph"], p["rainfall"], p["temperature"])
    assert "well matched" in tips[0].lower()
