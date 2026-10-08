# 💳 Credit Card Fraud Detection

## 🚀 Live Demo

👉 [Open Live Streamlit App]( https://credit-card-fraud-detection-nfmnhdev2dqrfcubf3amk9.streamlit.app)
 # 💳 Credit Card Fraud Detection

A Machine Learning web application that detects whether a credit card transaction is **Genuine** or **Fraudulent** using a **Random Forest Classifier**.

The model is integrated with a **Streamlit** interface where users can enter transaction details and receive a real-time prediction.

---

## 🚀 Project Overview

Credit card fraud is a major problem in financial transactions. This project uses Machine Learning to identify potentially fraudulent transactions based on transaction features.

The application:

- Accepts transaction details from the user
- Preprocesses the input using `StandardScaler`
- Uses a trained Random Forest model
- Predicts Genuine or Fraudulent transactions
- Displays prediction probabilities
- Shows risk level
- Displays important features used by the model

---

## 🧠 Machine Learning Model

**Algorithm:** Random Forest Classifier

**Preprocessing:** StandardScaler

**Input Features:** 30

The model uses:

- `Time`
- `V1` to `V28`
- `Amount`

**Target Variable:**

- `0` → Genuine Transaction
- `1` → Fraudulent Transaction

---

## 🖥️ Application Features

### 🔍 Fraud Prediction
Enter transaction details and get an immediate prediction.

### 📊 Prediction Probability
The application displays:

- Genuine probability
- Fraud probability

### 🚨 Risk Detection

- Genuine → Low Risk
- Fraud → High Risk

### 📈 Feature Importance
The application displays the top important features used by the Random Forest model.

### 🌙 Dark Mode
The Streamlit application supports Light and Dark mode.

### 📱 Interactive Dashboard
The application provides a simple and user-friendly interface for transaction analysis.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- StandardScaler
- Joblib
- Streamlit

---

## 📂 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── app.py
├── random_forest_fraud_model.pkl
├── fraud_scaler.pkl
├── requirements.txt
└── README.md