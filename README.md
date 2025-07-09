# 🪙 Gold Price Prediction using Machine Learning

This project aims to predict the price of gold based on various economic indicators using supervised machine learning models. The models are trained and tested on historical data to evaluate performance and accuracy.

---

## 📊 Problem Statement

Gold prices are influenced by various factors like stock market indices, currency exchange rates, and commodity prices. This project leverages these indicators to build a predictive model that estimates the gold price using ML algorithms.

---

## 📂 Dataset Features

- `SPX` – S&P 500 Index  
- `USO` – United States Oil Fund  
- `SLV` – Silver price  
- `EUR/USD` – Currency exchange rate  
- `GLD` – Gold ETF price (Target Variable)

---

## ⚙️ Technologies Used

- **Python**
- **Pandas, NumPy** – Data processing  
- **Matplotlib, Seaborn** – Data visualization  
- **Scikit-learn** – ML modeling  
- **Jupyter Notebook** – Development environment

---

## 🧠 ML Models Used

- **Linear Regression**
- **Random Forest Regressor**

---

## ✅ Workflow

1. **Data Cleaning & EDA**
   - Handled missing values
   - Visualized trends and correlations
2. **Feature Selection**
   - Selected economic indicators impacting gold price
3. **Model Building**
   - Trained and evaluated Linear Regression and Random Forest models
4. **Evaluation**
   - Compared using R² Score, RMSE, and Residual Plots
5. **Deployment Ready**
   - Model saved using Pickle for future use

---

## 📈 Results

- Random Forest Regressor showed higher accuracy than Linear Regression
- Captured non-linear patterns and gave better R² score on test data

---

## 📌 Project Highlights

- End-to-end regression model development
- Real-world financial dataset
- Model ready for deployment using Flask (optional future scope)

---

## 📁 Files in this Repository

- `gold_price_prediction.ipynb` – Full code notebook  
- `gld.csv` – Dataset  
- `model.pkl` – Trained model (if applicable)  
- `plots/` – Visualizations and result graphs  
- `README.md` – Project overview

---

