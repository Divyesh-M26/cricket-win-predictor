# 🏏 Cricket Win Predictor AI

A machine learning project that predicts cricket match winning probability using ball-by-ball data.

---

## 🎯 Problem Statement

In cricket, match outcomes are often judged intuitively during live games without structured data analysis.  
This project builds a machine learning system that predicts winning probability using real-time match situations derived from ball-by-ball data.

---

## 🚀 Features

- Ball-by-ball cricket analysis  
- Calculates:
  - Runs left  
  - Balls left  
  - Wickets left  
  - Current run rate  
  - Required run rate  
- Machine learning-based win probability prediction  
- Streamlit web interface for interactive use  

---

## 🧠 Why This Model?

Machine learning helps capture patterns in match situations that are not obvious through manual observation.

The model learns relationships between:
- Runs left  
- Balls left  
- Wickets remaining  
- Current run rate  
- Required run rate  

This allows prediction of dynamic win probability during a match.

---

## 🧠 Tech Stack

- Python  
- Pandas  
- Scikit-learn  
- Streamlit  
- Logistic Regression  

---

## 📊 How It Works

Feature Engineering:
- Runs left = Target − Current Score  
- Balls left = Remaining deliveries  
- Wickets left = Remaining wickets  
- Current Run Rate = Runs scored / Overs played  
- Required Run Rate = Runs left / Overs left  

---

## 🤖 Machine Learning Model

Algorithm: Logistic Regression  

Input Features:
- Runs left  
- Balls left  
- Wickets left  
- Current run rate  
- Required run rate  

Output:
- Win probability prediction  

---

## 📈 Key Insights

- Required Run Rate strongly affects match outcome  
- Wickets remaining are critical under pressure situations  
- Small changes in match situation can significantly change predictions  
- Logistic Regression provides interpretable probability output  

---

## ⚙️ How to Run

Install dependencies:

```bash
pip install pandas scikit-learn streamlit joblib

python -m streamlit run app.py

---

##📌 Future Improvements

Try advanced models (Random Forest, XGBoost)

Add live match data integration

Improve probability calibration

Add win probability visualization over time

---

##👨‍💻 Author

Divyesh M P

---

##🎯 Project Goal

To explore how machine learning can be applied to real-time sports situations by predicting win probability using structured match data.
