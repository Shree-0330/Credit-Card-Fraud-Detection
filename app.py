import streamlit as st
import pandas as pd
import joblib
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

@st.cache_resource
def load_model():

    model_path = "random_forest_fraud_model.pkl"
    scaler_path = "fraud_scaler.pkl"

    if not os.path.exists(model_path):
        st.error("random_forest_fraud_model.pkl not found.")
        st.stop()

    if not os.path.exists(scaler_path):
        st.error("fraud_scaler.pkl not found.")
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler


model, scaler = load_model()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("💳 Fraud Detection")

    st.markdown("---")

    dark_mode = st.toggle(
        "🌙 Dark Mode",
        value=False
    )

    st.markdown("---")

    st.subheader("📌 Model Information")

    st.write("**Algorithm:** Random Forest")
    st.write("**Preprocessing:** StandardScaler")
    st.write("**Features:** 30")
    st.write("**Target:** Fraud / Genuine")

# ============================================================
# LIGHT / DARK THEME
# ============================================================

if dark_mode:

    st.markdown("""
    <style>

    .stApp {
        background-color: #0f172a;
        color: white;
    }

    h1, h2, h3, h4, h5, p, label {
        color: white !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    </style>
    """, unsafe_allow_html=True)

else:

    st.markdown("""
    <style>

    .stApp {
        background-color: white;
        color: #111827;
    }

    h1, h2, h3, h4, h5, p, label {
        color: #111827 !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
    }

    </style>
    """, unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.title("💳 Credit Card Fraud Detection")

st.subheader(
    "🔍 Machine Learning Based Transaction Analysis"
)

st.write(
    "Enter the transaction details below to determine "
    "whether the transaction is Genuine or Fraudulent."
)

# ============================================================
# MODEL INFORMATION
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🤖 Algorithm", "Random Forest")

with col2:
    st.metric("📊 Features", "30")

with col3:
    st.metric("⚙️ Scaling", "StandardScaler")

with col4:
    st.metric("🎯 Output", "Fraud / Genuine")

# ============================================================
# TRANSACTION DETAILS
# ============================================================

st.markdown("---")

st.header("📝 Transaction Details")

col1, col2 = st.columns(2)

with col1:

    time = st.number_input(
        "⏱️ Transaction Time",
        value=0.0,
        format="%.4f"
    )

with col2:

    amount = st.number_input(
        "💰 Transaction Amount",
        min_value=0.0,
        value=100.0,
        format="%.2f"
    )

# ============================================================
# V1 - V28 FEATURES
# ============================================================

st.subheader("🔢 Transaction Features")

features = {}

columns = st.columns(4)

for i in range(1, 29):

    with columns[(i - 1) % 4]:

        features[f"V{i}"] = st.number_input(
            f"V{i}",
            value=0.0,
            format="%.6f",
            key=f"feature_{i}"
        )

# ============================================================
# PREDICT BUTTON
# ============================================================

st.markdown("---")

predict_button = st.button(
    "🔍 Predict Transaction",
    type="primary",
    use_container_width=True
)

# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # Create input dataframe

        input_data = pd.DataFrame([
            {
                "Time": time,
                **features,
                "Amount": amount
            }
        ])

        # Exact feature order used during training

        expected_columns = (
            ["Time"]
            + [f"V{i}" for i in range(1, 29)]
            + ["Amount"]
        )

        input_data = input_data[expected_columns]

        # Scale input

        input_scaled = scaler.transform(input_data)

        # Predict

        prediction = model.predict(input_scaled)[0]

        # Prediction probability

        probabilities = model.predict_proba(input_scaled)[0]

        genuine_probability = probabilities[0] * 100
        fraud_probability = probabilities[1] * 100

        # ====================================================
        # RESULT
        # ====================================================

        st.markdown("---")

        st.header("🚨 Prediction Result")

        if prediction == 1:

            st.error(
                "🚨 FRAUDULENT TRANSACTION DETECTED"
            )

            risk = "HIGH RISK"

        else:

            st.success(
                "✅ GENUINE TRANSACTION"
            )

            risk = "LOW RISK"

        # ====================================================
        # RESULT METRICS
        # ====================================================

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Genuine Probability",
                f"{genuine_probability:.2f}%"
            )

        with col2:

            st.metric(
                "Fraud Probability",
                f"{fraud_probability:.2f}%"
            )

        with col3:

            st.metric(
                "Risk Level",
                risk
            )

        # ====================================================
        # PROBABILITY CHART
        # ====================================================

        st.markdown("---")

        st.subheader("📊 Prediction Probability")

        probability_df = pd.DataFrame(
            {
                "Probability (%)": [
                    genuine_probability,
                    fraud_probability
                ]
            },
            index=[
                "Genuine",
                "Fraud"
            ]
        )

        st.bar_chart(
            probability_df,
            use_container_width=True
        )

        # ====================================================
        # TRANSACTION SUMMARY
        # ====================================================

        st.markdown("---")

        st.subheader("📋 Transaction Summary")

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f"**Transaction Time:** {time}"
            )

            st.write(
                f"**Transaction Amount:** ₹{amount:,.2f}"
            )

        with col2:

            result_text = (
                "Fraud"
                if prediction == 1
                else "Genuine"
            )

            st.write(
                f"**Prediction:** {result_text}"
            )

            st.write(
                f"**Risk Level:** {risk}"
            )

        # ====================================================
        # FEATURE IMPORTANCE
        # ====================================================

        if hasattr(model, "feature_importances_"):

            st.markdown("---")

            st.subheader("📈 Top 10 Important Features")

            importance_df = pd.DataFrame(
                {
                    "Feature": expected_columns,
                    "Importance": model.feature_importances_
                }
            )

            importance_df = (
                importance_df
                .sort_values(
                    "Importance",
                    ascending=False
                )
                .head(10)
            )

            importance_df = importance_df.set_index(
                "Feature"
            )

            st.bar_chart(
                importance_df,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"❌ Prediction Error: {e}"
        )

# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown("---")

st.header("⚙️ How It Works")

col1, col2, col3 = st.columns(3)

with col1:

    st.write("### 1️⃣ Input")

    st.write(
        "Enter transaction Time, Amount and "
        "V1 to V28 features."
    )

with col2:

    st.write("### 2️⃣ Preprocessing")

    st.write(
        "StandardScaler transforms the transaction "
        "data before prediction."
    )

with col3:

    st.write("### 3️⃣ Prediction")

    st.write(
        "Random Forest classifies the transaction "
        "as Genuine or Fraud."
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "💳 Credit Card Fraud Detection | "
    "Random Forest Machine Learning Project"
)