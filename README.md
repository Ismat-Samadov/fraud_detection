# TransactionDataGenerator

## Overview
`TransactionDataGenerator` is a Python-based tool for creating synthetic financial transaction datasets. It is designed to produce realistic, diverse transactions that can be used to train and test fraud detection models. Each dataset generated includes customer profiles, transaction attributes, merchant data, location, currency, and fraud indicators.

## Features
- **Customer Profiles**: Each customer has unique attributes like account age, card type, credit limit, spending patterns, and preferred devices.
- **Transaction Diversity**: Transactions cover various merchant categories, types, and include location, device, and currency details.
- **Fraud Simulation**: Transactions are generated with a specified percentage of fraudulent records, including factors like high-risk merchant categories, unusual amounts, and suspicious locations or devices.
- **Realistic Timestamps**: Each transaction has a timestamp that follows realistic shopping patterns and varies based on fraud likelihood.

## Requirements
- Python 3.x
- Required libraries: `pandas`, `numpy`, `datetime`, `hashlib`, `pytz`

Install required packages if necessary:
```bash
pip install pandas numpy pytz
```

## Usage

### Initialization and Dataset Generation
1. Clone or download this repository.
2. Run the script directly or use it as a module in other scripts. 

```python
import pandas as pd
import numpy as np
from transaction_data_generator import TransactionDataGenerator  # Adjust if necessary

# Initialize the generator
generator = TransactionDataGenerator()

# Generate dataset with parameters
df = generator.generate_dataset(
    num_customers=5000,
    transactions_per_customer=(50, 100),  # Range for number of transactions per customer
    fraud_percentage=0.2  # Set fraud percentage
)

# Save to CSV
df.to_csv("synthetic_fraud_data.csv", index=False)
print("Dataset saved to synthetic_fraud_data.csv")
```

### Parameters for `generate_dataset()`
- **`num_customers`**: Number of unique customer profiles.
- **`transactions_per_customer`**: Tuple specifying minimum and maximum number of transactions per customer (e.g., `(50, 100)`).
- **`fraud_percentage`**: Proportion of transactions to be labeled as fraudulent (e.g., `0.1` for 10% fraud rate).

### Example Output
A sample of the generated `synthetic_fraud_data.csv` file will contain columns such as:
- `transaction_id`: Unique identifier for the transaction.
- `customer_id`: Unique identifier for each customer.
- `merchant_category`: Category of the merchant (e.g., `Retail`, `Grocery`).
- `merchant_type`: Subcategory or specific type within the merchant category.
- `amount`: Transaction amount, adjusted by customer profile and currency.
- `currency`: Currency used in the transaction (e.g., USD, EUR).
- `is_fraud`: Boolean indicating if the transaction is fraudulent.

## Method Details

### Class: `TransactionDataGenerator`

#### `__init__()`
Initializes merchant, country, city, card, and device data used for transaction generation.

#### `generate_customer_profile()`
Creates a profile with attributes like account age, credit score, spending habits, and device preferences, which influence transaction behavior.

#### `generate_transaction()`
Generates a single transaction with customer-specific details and randomly assigned merchant, device, location, and fraud indicators.

#### `generate_dataset(num_customers, transactions_per_customer, fraud_percentage)`
Creates a dataset of transactions for the specified number of customers. It assigns fraudulent labels based on `fraud_percentage`.

#### Saving and Analyzing the Dataset
The generated dataset can be saved as a CSV file and used for training fraud detection algorithms or testing fraud scenarios.

## Example Script Output
```plaintext
Generated 250000 transactions
Fraud transactions: 50000
Dataset saved to synthetic_fraud_data.csv
```

## Use Cases
This tool is ideal for:
- **Machine Learning**: Training fraud detection models with synthetic transaction data.
- **Anomaly Detection**: Simulating financial data for testing anomaly detection algorithms.
- **Financial Analysis**: Creating realistic transaction datasets for data analysis exercises or educational purposes.