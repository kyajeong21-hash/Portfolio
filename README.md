# ✈️ Airline Ticket Price Prediction & Dashboard

> A time-series machine learning project predicting next-day ticket prices (One-period ahead) using market seat supply ratios and historical pricing data[cite: 1].

---

## 📌 Project Overview

Airline ticket prices fluctuate significantly based on departure dates, booking windows, competitor pricing, and market seat availability[cite: 1]. This project implements a **Random Forest Regressor incorporating market seat supply ratio assumptions** to forecast next-day ticket prices and visualizes price trends via an interactive web interface[cite: 1].

* **Deployment**: Streamlit Dashboard

---

## 🛠 Tech Stack

* **Language**: Python
* **Machine Learning**: Scikit-Learn (Random Forest Regressor), Pandas, NumPy
* **Visualization & Web App**: Streamlit, Plotly / Matplotlib

---

## ✨ Key Features

* **Interactive Filters**: Select Airline, Departure Date, and Sales Date via Streamlit UI.
* **One-Period Ahead Prediction**: Predicts ticket price for the day following the selected sales date ($t+1$)[cite: 1].
* **Trend Visualization**: Displays historical actual prices and model predictions side-by-side using **dashed line charts** for clear comparison.
* **Feature Integration**: Utilizes previous-day airline prices ($p_{i,t-1,a}$) and market seat supply ratios ($s_{i,t-1,a}$) as primary predictive features[cite: 1].

---

## 📊 Model Performance & Insights

**Market Seat Supply Ratio Model (Model 1)** Results[cite: 1]:

* **Test RMSE**: `₩43,318`[cite: 1]
* **Test $R^2$**: `0.9209` (92.1% variance explained)[cite: 1]
* **Top Feature Importances**[cite: 1]:
  1. Previous-day Jin Air (LJ) price ($31.1\%$)[cite: 1]
  2. Jin Air (LJ) dummy variable ($13.6\%$)[cite: 1]
  3. Previous-day T'way Air (TW) price ($11.3\%$)[cite: 1]
  4. Jeju Air (7C) dummy variable ($9.2\%$)[cite: 1]

---

## 🔮 Future Improvements

1. **Extended Forecast Horizon ($\tau$-period ahead)**: Expand beyond 1-day predictions ($t+1$) to forecast farther into the future ($t+\tau$)[cite: 1].
2. **Adaptive Rolling Residual Correction**: Track prediction errors dynamically on a rolling basis to apply direct error-correction adjustments[cite: 1].
