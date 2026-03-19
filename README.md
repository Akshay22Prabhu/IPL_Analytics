# 🏏 IPL Analytics Platform

An end-to-end machine learning and AI-powered analytics platform for predicting IPL player performance, forecasting future runs, and generating AI-driven insights.

---

## 🚀 Features

* 🔮 **Run Prediction Model** – Predict player runs using ML (XGBoost)
* 📈 **Time Series Forecasting** – Predict next 10 match performances
* 🤖 **AI Cricket Analyst** – Generate player insights using LLMs
* 📊 **Interactive Dashboard** – Built using Streamlit
* ⚔️ **Player Comparison** – Compare performance metrics
* 📉 **EDA Notebook** – Detailed exploratory data analysis

---

## 🧠 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Prophet
* Streamlit
* HuggingFace Transformers

---

## 📂 Project Structure

```
app/                # Streamlit UI
src/                # Core ML + data pipeline
data/               # Dataset (processed)
models/             # Trained models
notebooks/          # EDA
```

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```
git clone https://github.com/<your-username>/ipl-analytics.git
cd ipl-analytics
```

### 2. Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Run Pipeline

```
python src/feature_engineering.py
python src/train_model.py
```

### 5. Run App

```
streamlit run app/streamlit_app.py
```

---

## 📊 Sample Outputs

* Player run prediction
* Future match forecasts
* AI-generated performance insights

---

## 🧪 EDA

Notebook available in:

```
notebooks/eda.ipynb
```

---

## 🎯 Key Highlights

* Implemented advanced feature engineering (rolling averages, consistency, trend)
* Built ML pipeline using XGBoost with categorical encoding
* Integrated time-series forecasting using Prophet
* Developed GenAI-based insights module using LLMs

---

## 🚀 Future Improvements

* Player vs opponent modeling
* Fantasy team recommendation system
* Deployment on cloud (AWS / Streamlit Cloud)
* REST API using FastAPI

---

## 📄 License

This project is for educational and portfolio purposes.
