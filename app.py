import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Load Model Files
# -----------------------------

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
threshold = joblib.load("threshold.pkl")


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# -----------------------------
# Custom Styling
# -----------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }

    .metric-box {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Header
# -----------------------------

st.markdown(
    '<div class="main-title">💳 Credit Card Fraud Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning powered transaction risk analysis'
    '</div>',
    unsafe_allow_html=True
)


st.divider()


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("⚙️ Model Information")

st.sidebar.write(
    "This application uses a Logistic Regression "
    "model trained for credit card fraud detection."
)

st.sidebar.metric(
    "Decision Threshold",
    f"{threshold:.2f}"
)

st.sidebar.info(
    "A transaction is classified as fraudulent when "
    "its predicted fraud probability is greater than "
    "or equal to the decision threshold."
)


# -----------------------------
# Transaction Details
# -----------------------------

st.header("💰 Transaction Information")

col1, col2 = st.columns(2)

with col1:
    time = st.number_input(
        "Transaction Time",
        value=0.0
    )

with col2:
    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0
    )


# -----------------------------
# V1 - V28 Features
# -----------------------------

st.header("🔢 Transaction Features")

st.write(
    "Enter the anonymized transaction features used by the trained model."
)


features = {}

cols = st.columns(4)

for i in range(1, 29):

    with cols[(i - 1) % 4]:

        features[f"V{i}"] = st.number_input(
            f"V{i}",
            value=0.0,
            format="%.6f"
        )


# -----------------------------
# Prediction
# -----------------------------

st.divider()

predict_button = st.button(
    "🔍 Analyze Transaction",
    use_container_width=True
)


if predict_button:

    # Create input dataframe
    input_data = pd.DataFrame([{
        "Time": time,
        **features,
        "Amount": amount
    }])

    # Ensure exact feature order
    input_data = input_data[
        ["Time"] +
        [f"V{i}" for i in range(1, 29)] +
        ["Amount"]
    ]

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Fraud probability
    probability = model.predict_proba(
        input_scaled
    )[0][1]

    # Prediction using threshold
    prediction = int(probability >= threshold)


    # -----------------------------
    # Results
    # -----------------------------

    st.header("📊 Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Decision Threshold",
            f"{threshold * 100:.0f}%"
        )

    with col3:

        if prediction == 1:
            st.metric(
                "Risk Status",
                "🚨 FRAUD"
            )
        else:
            st.metric(
                "Risk Status",
                "✅ LEGITIMATE"
            )


    # -----------------------------
    # Result Message
    # -----------------------------

    if prediction == 1:

        st.error(
            "🚨 FRAUDULENT TRANSACTION DETECTED"
        )

        st.write(
            f"The model estimates a fraud probability of "
            f"**{probability * 100:.2f}%**, which is above "
            f"the decision threshold of "
            f"**{threshold * 100:.0f}%**."
        )

    else:

        st.success(
            "✅ TRANSACTION APPEARS LEGITIMATE"
        )

        st.write(
            f"The model estimates a fraud probability of "
            f"**{probability * 100:.2f}%**, which is below "
            f"the decision threshold of "
            f"**{threshold * 100:.0f}%**."
        )


    # -----------------------------
    # Transaction Summary
    # -----------------------------

    st.subheader("📋 Transaction Summary")

    summary = pd.DataFrame({
        "Parameter": [
            "Transaction Time",
            "Transaction Amount",
            "Fraud Probability",
            "Decision Threshold",
            "Final Prediction"
        ],
        "Value": [
            time,
            f"${amount:.2f}",
            f"{probability * 100:.4f}%",
            f"{threshold * 100:.0f}%",
            "Fraud" if prediction == 1 else "Legitimate"
        ]
    })

    st.table(summary)


# -----------------------------
# Model Performance
# -----------------------------

st.divider()

st.header("📈 Model Performance")

st.write(
    "Performance metrics calculated on the Credit Card Fraud Detection dataset."
)

# Load dataset
evaluation_data = pd.read_csv("creditcard.csv")

X_eval = evaluation_data.drop("Class", axis=1)
y_eval = evaluation_data["Class"]

# Scale features
X_eval_scaled = scaler.transform(X_eval)

# Predictions
evaluation_probability = model.predict_proba(
    X_eval_scaled
)[:, 1]

evaluation_prediction = (
    evaluation_probability >= threshold
).astype(int)


# Calculate metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

accuracy = accuracy_score(
    y_eval,
    evaluation_prediction
)

precision = precision_score(
    y_eval,
    evaluation_prediction
)

recall = recall_score(
    y_eval,
    evaluation_prediction
)

f1 = f1_score(
    y_eval,
    evaluation_prediction
)


# Display metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

with col2:
    st.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )

with col3:
    st.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )

with col4:
    st.metric(
        "F1 Score",
        f"{f1 * 100:.2f}%"
    )

# -----------------------------
# Confusion Matrix
# -----------------------------

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_eval,
    evaluation_prediction
)

st.subheader("🔎 Confusion Matrix")

cm_data = pd.DataFrame(
    cm,
    index=["Actual Legitimate", "Actual Fraud"],
    columns=["Predicted Legitimate", "Predicted Fraud"]
)

st.dataframe(
    cm_data,
    use_container_width=True
)

# -----------------------------
# Class Distribution
# -----------------------------

st.subheader("📊 Transaction Class Distribution")

legitimate_count = int((y_eval == 0).sum())
fraud_count = int((y_eval == 1).sum())

distribution_data = pd.DataFrame({
    "Transaction Type": [
        "Legitimate",
        "Fraud"
    ],
    "Count": [
        legitimate_count,
        fraud_count
    ]
})

st.bar_chart(
    distribution_data.set_index("Transaction Type")
)
    # -----------------------------
# Batch CSV Prediction
# -----------------------------

st.divider()

st.header("📁 Batch Fraud Detection")

st.write(
    "Upload a CSV file containing transactions to analyze multiple "
    "transactions at once."
)

uploaded_file = st.file_uploader(
    "Upload Transaction CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        batch_data = pd.read_csv(uploaded_file)

        required_columns = (
            ["Time"] +
            [f"V{i}" for i in range(1, 29)] +
            ["Amount"]
        )

        missing_columns = [
            col for col in required_columns
            if col not in batch_data.columns
        ]

        if missing_columns:

            st.error(
                "The uploaded CSV is missing these columns: "
                + ", ".join(missing_columns)
            )

        else:

            # Select features in exact model order
            batch_features = batch_data[
                required_columns
            ]

            # Scale features
            batch_scaled = scaler.transform(
                batch_features
            )

            # Calculate fraud probability
            batch_probability = model.predict_proba(
                batch_scaled
            )[:, 1]

            # Generate predictions
            batch_prediction = (
                batch_probability >= threshold
            ).astype(int)

            # Add results
            result_data = batch_data.copy()

            result_data["Fraud Probability"] = (
                batch_probability
            )

            result_data["Prediction"] = batch_prediction

            result_data["Risk Status"] = result_data[
                "Prediction"
            ].map({
                0: "Legitimate",
                1: "Fraud"
            })

            # Summary
            total_transactions = len(result_data)

            fraud_transactions = int(
                result_data["Prediction"].sum()
            )

            legitimate_transactions = (
                total_transactions -
                fraud_transactions
            )

            st.subheader("📊 Batch Analysis Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Total Transactions",
                    total_transactions
                )

            with col2:
                st.metric(
                    "Fraudulent Transactions",
                    fraud_transactions
                )

            with col3:
                st.metric(
                    "Legitimate Transactions",
                    legitimate_transactions
                )

            # Show results
            st.subheader("🔍 Transaction Results")

            st.dataframe(
                result_data,
                use_container_width=True
            )

            # Download results
            csv_result = result_data.to_csv(
                index=False
            )

            st.download_button(
                label="📥 Download Results",
                data=csv_result,
                file_name="fraud_detection_results.csv",
                mime="text/csv"
            )

    except Exception as e:

        st.error(
            f"Error processing the file: {e}"
        )