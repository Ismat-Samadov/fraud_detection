# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
import gc



# Load data
train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')
sample_submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')

# Merge transaction and identity datasets
train = train_transaction.merge(train_identity, on='TransactionID', how='left')
test = test_transaction.merge(test_identity, on='TransactionID', how='left')

# Clean up memory
del train_transaction, train_identity, test_transaction, test_identity
gc.collect()

# Feature Engineering (basic example)
# Label encoding for categorical variables
for col in train.select_dtypes(include=['object']).columns:
    if col in test.columns:
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))

# Fill missing values with -1 (or you can try other imputation methods)
train = train.fillna(-1)
test = test.fillna(-1)

# Separate target and features
X = train.drop(['isFraud', 'TransactionID'], axis=1)
y = train['isFraud']
X_test = test.drop(['TransactionID'], axis=1)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize LightGBM model
model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=64,
    colsample_bytree=0.8,
    subsample=0.8,
    max_depth=12,
    random_state=42
)

# Train the model with callbacks for early stopping and logging
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_metric='auc',
    callbacks=[
        early_stopping(stopping_rounds=100),
        log_evaluation(100)  # Logs every 100 rounds
    ]
)

# Predict on validation set and evaluate
val_preds = model.predict_proba(X_val)[:, 1]
roc_score = roc_auc_score(y_val, val_preds)
print(f'Validation ROC-AUC score: {roc_score}')

# Predict on test set for submission
test_preds = model.predict_proba(X_test)[:, 1]

# Create submission file
sample_submission['isFraud'] = test_preds
sample_submission.to_csv('submission.csv', index=False)
