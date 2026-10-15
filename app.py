"""Streamlit app: enter soil & weather values, get crop suggestions and soil advice.

Run:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from src.config import CROP_PROFILES, FEATURE_LABELS, FIGURES, MODEL_PATH
from src.recommend import CropRecommender

st.set_page_config(page_title="Crop Recommendation System", page_icon="🌾", layout="wide")
st.title("🌾 Smart Crop Recommendation")
st.caption("Enter your soil test values and local weather to find the best-suited crops and get soil-health advice.")

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python -m src.generate_data` and `python -m src.train` first.")
    st.stop()


@st.cache_resource
def load_recommender() -> CropRecommender:
    return CropRecommender()


rec = load_recommender()
tab_rec, tab_crops, tab_model = st.tabs(["Recommend", "Crop guide", "About the model"])

with tab_rec:
    left, right = st.columns([1, 1.2])
    with left:
        st.subheader("Field conditions")
        N = st.slider(FEATURE_LABELS["N"], 0, 140, 80)
        P = st.slider(FEATURE_LABELS["P"], 5, 145, 45)
        K = st.slider(FEATURE_LABELS["K"], 5, 205, 40)
        temperature = st.slider(FEATURE_LABELS["temperature"], 8.0, 44.0, 25.0, 0.5)
        humidity = st.slider(FEATURE_LABELS["humidity"], 14.0, 100.0, 70.0, 1.0)
        ph = st.slider(FEATURE_LABELS["ph"], 3.5, 9.5, 6.5, 0.1)
        rainfall = st.slider(FEATURE_LABELS["rainfall"], 20.0, 300.0, 100.0, 5.0)

    result = rec.recommend(N, P, K, temperature, humidity, ph, rainfall, k=3)
    top = result["top_crops"]
    with right:
        best = top.iloc[0]
        st.subheader("Recommendation")
        st.metric("Best crop", best["crop"], f"{best['confidence']:.0%} confidence")
        st.caption(f"Season: {best['season']}")
        st.bar_chart(top.set_index("crop")["confidence"])
        st.subheader(f"Soil & climate advice for {best['crop']}")
        for tip in result["advice"]:
            st.write(f"- {tip}")
        if best["confidence"] < 0.6:
            st.info("Confidence is moderate — conditions suit several crops. Consider the runners-up and local market prices.")

with tab_crops:
    guide = pd.DataFrame(CROP_PROFILES).T.rename_axis("crop").reset_index()
    guide = guide.rename(columns={"ph": "pH", "temperature": "temp (°C)", "humidity": "humidity (%)", "rainfall": "rainfall (mm)"})
    st.dataframe(guide, use_container_width=True, hide_index=True)
    st.caption("Approximate ideal growing conditions used to build the model and generate advice.")

with tab_model:
    for name in ["feature_importance.png", "confusion_matrix.png"]:
        if (FIGURES / name).exists():
            st.image(str(FIGURES / name))
    st.warning("Decision-support only. Validate recommendations with local soil testing and your agriculture extension officer.")
