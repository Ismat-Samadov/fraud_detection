from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('fraud_detection_model.pkl')

# Define a prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Parse JSON request data
    data = request.get_json()
    
    # Convert the data into a DataFrame (ensuring the structure matches the training data)
    input_data = pd.DataFrame(data, index=[0])
    
    # Make prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0]  # Get the probabilities for fraud and non-fraud
    
    # Interpret prediction and probabilities
    result = {
        'prediction': 'Fraudulent Transaction' if prediction[0] == 1 else 'Non-Fraudulent Transaction',
        'probability_of_non_fraud': probability[0],
        'probability_of_fraud': probability[1]
    }
    
    return jsonify(result)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
