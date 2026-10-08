import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = "random_forest_fraud_model.pkl"
    scaler_path = "fraud_scaler.pkl"

    if not os.path.exists(model_path):
        st.error(f"❌ {model_path} not found.")
        st.stop()

    if not os.path.exists(scaler_path):
        st.error(f"❌ {scaler_path} not found.")
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

    st.subheader("🎨 Appearance")

    dark_mode = st.toggle(
        "🌙 Dark Mode",
        value=False
    )

    st.markdown("---")

    st.subheader("📌 Model")

    st.info(
        """
        **Algorithm:** Random Forest

        **Preprocessing:** StandardScaler

        **Features:** 30

        **Target:** Fraud / Genuine
        """
    )

    st.markdown("---")

    st.caption("Credit Card Fraud Detection System")
    st.caption("Machine Learning Project")


# ============================================================
# THEME
# ============================================================

if dark_mode:

    background = "#0f172a"
    card = "#1e293b"
    text = "#f8fafc"
    secondary = "#94a3b8"
    border = "#334155"
    input_bg = "#1e293b"

else:

    background = "#ffffff"
    card = "#ffffff"
    text = "#111827"
    secondary = "#6b7280"
    border = "#e5e7eb"
    input_bg = "#ffffff"


st.markdown(
    f"""
    <style>

    .stApp {{
        background-color: {background};
        color: {text};
    }}

    .block-container {{
        max-width: 1400px;
        padding-top: 2rem;
    }}

    h1, h2, h3, h4, p, label {{
        color: {text} !important;
    }}

    section[data-testid="stSidebar"] {{
        background-color: {card};
        border-right: 1px solid {border};
    }}

    section[data-testid="stSidebar"] * {{
        color: {text} !important;
    }}

    .dashboard-card {{
        background-color: {card};
        border: 1px solid {border};
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }}

    .metric-title {{
        color: {secondary} !important;
        font-size: 14px;
        font-weight: 600;
    }}

    .metric-value {{
        color: {text} !important;
        font-size: 28px;
        font-weight: 700;
    }}

    .prediction-box {{
        background-color: {card};
        border: 2px solid {border};
        border-radius: 18px;
        padding: 30px;
        text-align: center;
        margin-top: 20px;
    }}

    input {{
        background-color: {input_bg} !important;
        color: {text} !important;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.title("💳 Credit Card Fraud Detection")

st.markdown(
    "### 🔍 Machine Learning Based Transaction Analysis"
)

st.markdown(
    "Enter the transaction details below to determine whether the transaction is **Genuine** or **Fraudulent**."
)


# ============================================================
# MODEL INFORMATION CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div class="dashboard-card">
        <div class="metric-title">🤖 Algorithm</div>
        <div class="metric-value">Random Forest</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="dashboard-card">
        <div class="metric-title">📊 Features</div>
        <div class="metric-value">30</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="dashboard-card">
        <div class="metric-title">⚙️ Scaling</div>
        <div class="metric-value">StandardScaler</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        """
        <div class="dashboard-card">
        <div class="metric-title">🎯 Output</div>
        <div class="metric-value">Fraud / Genuine</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TRANSACTION INPUT
# ============================================================

st.markdown("---")

st.header("📝 Transaction Details")

st.info(
    "Enter the transaction values used by the trained model."
)


# ============================================================
# TIME + AMOUNT
# ============================================================

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
# V1 - V28
# ============================================================

st.subheader("🔢 PCA Features")

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
# BUTTONS
# ============================================================

st.markdown("---")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:

    predict_button = st.button(
        "🔍 Predict Transaction",
        use_container_width=True,
        type="primary"
    )

with col2:

    reset_button = st.button(
        "🔄 Reset",
        use_container_width=True
    )


# ============================================================
# RESET
# ============================================================

if reset_button:

    st.rerun()


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # Create input DataFrame

        input_data = pd.DataFrame([
            {
                "Time": time,
                **features,
                "Amount": amount
            }
        ])

        # Exact feature order

        expected_columns = (
            ["Time"]
            + [f"V{i}" for i in range(1, 29)]
            + ["Amount"]
        )

        input_data = input_data[expected_columns]


        # Scale input

        input_scaled = scaler.transform(input_data)


        # Prediction

        prediction = model.predict(input_scaled)[0]


        # Probability

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
        # KPI CARDS
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

        probability_df = pd.DataFrame({
            "Transaction Type": [
                "Genuine",
                "Fraud"
            ],
            "Probability": [
                genuine_probability,
                fraud_probability
            ]
        })


        fig = px.bar(
            probability_df,
            x="Transaction Type",
            y="Probability",
            text="Probability",
            title="Transaction Classification Probability"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig.update_layout(
            yaxis_title="Probability (%)",
            xaxis_title="",
            yaxis_range=[0, 100],
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # ====================================================
        # TRANSACTION SUMMARY
        # ====================================================

        st.markdown("---")

        st.subheader("📋 Transaction Summary")

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:

            st.write(
                f"**Transaction Time:** {time}"
            )

            st.write(
                f"**Transaction Amount:** ₹{amount:,.2f}"
            )

        with summary_col2:

            st.write(
                f"**Prediction:** "
                f"{'Fraud' if prediction == 1 else 'Genuine'}"
            )

            st.write(
                f"**Risk:** {risk}"
            )


        # ====================================================
        # FEATURE IMPORTANCE
        # ====================================================

        if hasattr(model, "feature_importances_"):

            st.markdown("---")

            st.subheader("📈 Feature Importance")

            importance_df = pd.DataFrame({

                "Feature": expected_columns,

                "Importance":
                    model.feature_importances_

            })

            importance_df = (
                importance_df
                .sort_values(
                    "Importance",
                    ascending=False
                )
                .head(10)
            )


            fig2 = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 10 Important Features"
            )

            fig2.update_layout(
                height=500
            )

            st.plotly_chart(
                fig2,
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

    st.markdown(
        """
        ### 1️⃣ Input

        Transaction details are entered
        into the application.
        """
    )

with col2:

    st.markdown(
        """
        ### 2️⃣ Preprocessing

        The input is transformed using
        the trained StandardScaler.
        """
    )

with col3:

    st.markdown(
        """
        ### 3️⃣ Prediction

        Random Forest predicts whether
        the transaction is Genuine or Fraud.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "💳 Credit Card Fraud Detection | Random Forest Machine Learning Project"
)