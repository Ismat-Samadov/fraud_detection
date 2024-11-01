# Transaction Fraud Detection Project

## Overview

During my research on fraud detection models, I faced several data challenges that prompted the need for a custom data generation tool. Many available open-source datasets, particularly in financial domains, feature encoded variable names like `v1`, `v2`, ... `v22`, which obscure the nature of each feature. This lack of interpretability makes it difficult to identify and explain which features are significant in detecting fraud, limiting model transparency and deployment potential.

To overcome these limitations, I developed a synthetic banking data generator that produces realistic transaction data with meaningful feature names, such as `merchant_name` instead of an anonymous code like `v2`. This approach facilitates an end-to-end fraud detection model with enhanced interpretability and practical applications. The custom generator allows control over the number of customer transactions, transaction range, and the fraud percentage, making it ideal for understanding fraud detection workflows and exploring model performance in a controlled environment.

---

## Project Structure

```plaintext
.
├── README.md                    # Documentation of project setup, structure, usage, and details
├── app.py                       # Flask app for deploying the fraud detection model as a web service
├── data_gen.py                  # Script for generating synthetic transaction data
├── fraud_detection_model.pkl     # Trained and saved fraud detection model
├── model.ipynb                  # Jupyter notebook for data generation, model training, and evaluation
├── requirements.txt             # List of Python libraries required for the project
├── static                       # Directory for static files (CSS)
│   └── style.css                # CSS styling for the web application interface
├── synthetic_fraud_data.csv     # Generated synthetic dataset for model training/testing
└── templates                    # Directory for HTML templates
    └── index.html               # HTML template for the web application interface
```

### Explanation of Project Files

- **README.md**: Project documentation, including setup, structure, usage, and performance details.
- **app.py**: Flask web application that loads the model, accepts transaction data, and returns predictions for fraud detection.
- **data_gen.py**: Script to generate synthetic transaction data. This enables custom datasets with realistic attributes and control over fraud prevalence.
- **fraud_detection_model.pkl**: Pre-trained fraud detection model saved using `joblib`.
- **model.ipynb**: Notebook that contains the end-to-end workflow, from data generation and preprocessing to model training, evaluation, and saving.
- **requirements.txt**: Lists required packages (e.g., `pandas`, `flask`) for setting up the environment.
- **static/style.css**: CSS file for styling the web interface, ensuring a responsive, user-friendly design.
- **synthetic_fraud_data.csv**: Sample dataset generated by `data_gen.py`, used for training and testing the model.
- **templates/index.html**: HTML template for the web application’s user interface, allowing input of transaction data and displaying predictions.

---

## Data Generation

**Why Synthetic Data Generation?**
The data generator was developed to create realistic, interpretable financial transaction datasets. Instead of dealing with obscure feature names (`v1`, `v2`), the generator produces descriptive attributes (e.g., `merchant_category`, `device`) that clarify each feature's role in fraud detection. This interpretability improves both the model-building process and user experience when deploying fraud detection applications.

### Key Features of the Data Generator

- **Customer Profiles**: Simulates customer behavior using attributes like credit score, spending patterns, and device preferences.
- **Transaction Attributes**: Provides diverse transaction details, covering merchant type, location, and currency.
- **Fraud Simulation**: Allows specification of a fraud percentage, creating realistic fraud scenarios with high-risk patterns.
- **Realistic Timestamps**: Adds transaction timestamps to mimic real shopping behavior.

### Example Data Generation

To generate a dataset, specify the number of customers, range of transactions, and fraud percentage:

```python
if __name__ == "__main__":
    generator = TransactionDataGenerator()
    df = generator.generate_dataset(
        num_customers=500,
        transactions_per_customer=(100, 200),
        fraud_percentage=0.1
    )
    df.to_csv("synthetic_fraud_data.csv", index=False)
    print("Dataset saved to synthetic_fraud_data.csv")
```

This command will create a dataset with 500 customers, each having 100 to 200 transactions, and 10% of transactions labeled as fraudulent.

---

## Model Performance

The fraud detection model trained on this dataset demonstrates robust performance, achieving high precision and recall in identifying fraud.

**Classification Report**:
| Metric         | Class        | Precision | Recall | F1-Score | Support |
|----------------|--------------|-----------|--------|----------|---------|
| **Non-Fraud** | False (0)    | 0.98      | 1.00   | 0.99     | 20,308  |
| **Fraud**     | True (1)     | 0.96      | 0.77   | 0.86     | 2,186   |
| **Accuracy**  |              | 0.97      |        |          | 22,494  |
| **Macro Avg** |              | 0.97      | 0.88   | 0.92     |         |
| **Weighted Avg** |          | 0.97      | 0.97   | 0.97     |         |

- **ROC-AUC Score**: 0.88

The high ROC-AUC score and precision metrics underscore the model’s effectiveness in distinguishing fraudulent from non-fraudulent transactions, making it a reliable solution for fraud detection.

---

## Model Prediction Example

To make predictions using the trained model, load it, prepare sample transaction data, and interpret the prediction as shown:

```python
# Import necessary libraries
import joblib
import pandas as pd

# Load the saved model
model = joblib.load('fraud_detection_model.pkl')

# Sample input data for prediction
new_data = pd.DataFrame({
    'amount': [500.0],
    'distance_from_home': [1],
    'transaction_hour': [13],
    'merchant_category': ['Retail'],
    'merchant_type': ['online'],
    'merchant': ['Amazon'],
    'currency': ['USD'],
    'country': ['United States'],
    'city': ['New York'],
    'city_size': ['large'],
    'card_type': ['debit'],
    'device': ['Chrome'],
    'channel': ['web'],
    'day_of_week': [2],
    'is_weekend': [0],
    'num_transactions_last_hour': [3],
    'total_amount_last_hour': [1500.0],
})

# Make prediction
prediction = model.predict(new_data)

# Interpret prediction
if prediction[0] == 1:
    print("Prediction: Fraudulent Transaction")
else:
    print("Prediction: Non-Fraudulent Transaction")
    
# Get probabilities for each class
proba = model.predict_proba(new_data)

print(f"Probability of Non-Fraudulent Transaction: {proba[0][0]:.2f}")
print(f"Probability of Fraudulent Transaction: {proba[0][1]:.2f}")
```

### Interpretation
- **Prediction**: The model returns whether a transaction is likely fraudulent or not.
- **Probability**: Probabilities for each class provide insight into prediction confidence, allowing for risk-based thresholds in fraud detection.

This example demonstrates how to leverage the trained model in real-world applications, offering both prediction and confidence levels for fraud detection.

