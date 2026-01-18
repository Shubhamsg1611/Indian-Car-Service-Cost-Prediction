import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Taxi Trip Price Prediction",
    layout="centered"
)

st.title("🚕 Taxi Trip Price Prediction")
st.write("Enter trip details to predict the estimated trip price.")

# --------------------------------------------------
# Load Model Artifacts
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("trip_price_model.joblib")
    numeric_features = joblib.load("numeric_features.joblib")
    categorical_features = joblib.load("categorical_features.joblib")
    target = joblib.load("target_variable.joblib")
    return model, numeric_features, categorical_features, target

model, numeric_features, categorical_features, target = load_artifacts()

# --------------------------------------------------
# User Input Section
# --------------------------------------------------
st.subheader("📥 Trip Details")

input_data = {}

# -------- Numeric Inputs --------
for feature in numeric_features:
    input_data[feature] = st.number_input(
        label=feature,
        value=1.0,
        min_value=0.0,
        step=0.1
    )

# -------- Categorical Inputs --------
for feature in categorical_features:
    input_data[feature] = st.selectbox(
        label=feature,
        options=["Morning", "Afternoon", "Evening", "Night",
                 "Weekday", "Weekend",
                 "Low", "Medium", "High",
                 "Sunny", "Rainy", "Foggy",
                 "Economy", "Standard", "Premium", "Luxury"]
    )

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔮 Predict Trip Price"):
    try:
        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)[0]

        st.success(f"💰 Predicted {target}: ₹ {round(prediction, 2)}")

    except Exception as e:
        st.error("Prediction failed. Please check inputs.")
        st.exception(e)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Machine Learning powered Taxi Fare Prediction")