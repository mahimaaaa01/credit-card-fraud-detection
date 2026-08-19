# 💳 Credit Card Fraud Detection System

## 📌 Project Overview

This project is a Machine Learning based Credit Card Fraud Detection System that identifies potentially fraudulent transactions using Logistic Regression.

The system provides an interactive Streamlit dashboard for analyzing individual transactions as well as multiple transactions through CSV batch processing.

---

## 🚀 Features

- Individual transaction fraud prediction
- Fraud probability calculation
- Custom decision threshold
- Batch CSV fraud detection
- Model performance evaluation
- Confusion matrix
- Transaction class distribution visualization
- Downloadable prediction results
- Interactive Streamlit dashboard

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Logistic Regression
- StandardScaler
- Streamlit
- Joblib

---

## 📊 Dataset

The project uses the Credit Card Fraud Detection dataset.

### Dataset Details

- Total transactions: **284,807**
- Legitimate transactions: **284,315**
- Fraudulent transactions: **492**
- Input features: **30**
- Features: `Time`, `V1`–`V28`, `Amount`
- Target variable: `Class`

Where:

- `Class = 0` → Legitimate transaction
- `Class = 1` → Fraudulent transaction

---

## 🤖 Machine Learning Model

The project uses **Logistic Regression** for binary classification.

### Preprocessing

The transaction features are standardized using `StandardScaler`.

### Decision Threshold

A decision threshold of **0.99** is used for fraud classification.

If:

```text
Fraud Probability >= 0.99
```

the transaction is classified as:

```text
Fraud
```

Otherwise, it is classified as:

```text
Legitimate
```

---

## 📈 Model Performance

The model was evaluated on the complete dataset.

| Metric | Score |
|---|---:|
| Accuracy | 99.88% |
| Precision | 62.01% |
| Recall | 82.93% |
| F1 Score | 70.96% |

### Confusion Matrix

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 284,065 | 250 |
| Actual Fraud | 84 | 408 |

The model correctly detected **408 fraudulent transactions** while missing **84 fraudulent transactions**.

---

## 🖥️ Application

The Streamlit application allows users to:

1. Enter transaction details.
2. Analyze the transaction using the trained model.
3. View the predicted fraud probability.
4. Compare the probability with the decision threshold.
5. Receive a Legitimate or Fraud classification.
6. Upload a CSV file for batch fraud detection.
7. Download the prediction results.

---

## 📂 Project Structure

```text
Credit_Card_Fraud_Detection/
│
├── app.py
├── fraud_model.pkl
├── scaler.pkl
├── threshold.pkl
├── creditcard.csv
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

### 1. Install the required libraries

Open the terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit application

```bash
python -m streamlit run app.py
```

### 3. Open the application

Streamlit will provide a local URL such as:

```text
http://localhost:8501
```

Open this URL in your browser.

---

## 📁 Batch Prediction

The application supports CSV batch prediction.

Upload a CSV containing the following features:

```text
Time
V1
V2
V3
...
V28
Amount
```

The application generates:

- Fraud Probability
- Prediction
- Risk Status

The results can be downloaded as a CSV file.

---

## 🎯 Future Improvements

- Experiment with Random Forest and XGBoost
- Hyperparameter tuning
- ROC-AUC analysis
- Precision-Recall curves
- Model explainability using SHAP
- Real-time transaction monitoring
- Cloud deployment
- API integration

---

## 👩‍💻 Author

**Mahima Verma**

B.Tech Electronics and Computer Engineering Student

Interested in Data Science, Machine Learning, Artificial Intelligence and Software Development.
