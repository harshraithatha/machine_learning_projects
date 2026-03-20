from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "best_model.joblib"


st.set_page_config(page_title="Sales Prediction App")
st.title("Sales Prediction App")
st.write("Enter a few business inputs and estimate expected sales.")


if not MODEL_PATH.exists():
    st.error("Train the project first with `python projects/05_sales_prediction_app/train.py`.")
else:
    model = joblib.load(MODEL_PATH)

    advertising_spend = st.number_input("Advertising spend", min_value=0.0, value=2500.0, step=100.0)
    store_visits = st.number_input("Store visits", min_value=0, value=400, step=10)
    discount_pct = st.slider("Discount percent", min_value=0.0, max_value=50.0, value=10.0, step=0.5)
    holiday_flag = st.selectbox("Holiday period", options=[0, 1], format_func=lambda x: "Yes" if x else "No")
    competitor_price_index = st.slider(
        "Competitor price index",
        min_value=0.7,
        max_value=1.3,
        value=1.0,
        step=0.01,
    )
    season = st.selectbox("Season", options=["Winter", "Spring", "Summer", "Autumn"])

    input_frame = pd.DataFrame(
        {
            "advertising_spend": [advertising_spend],
            "store_visits": [store_visits],
            "discount_pct": [discount_pct],
            "holiday_flag": [holiday_flag],
            "competitor_price_index": [competitor_price_index],
            "season": [season],
        }
    )

    if st.button("Predict sales"):
        prediction = model.predict(input_frame)[0]
        st.success(f"Predicted sales: {prediction:,.2f}")
