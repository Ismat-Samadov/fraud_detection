from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('fraud_detection_model.pkl')

# Home route to render the form
@app.route('/')
def index():
    return render_template('index.html')

# Prediction API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data and convert to DataFrame
    data = {
        'amount': float(request.form['amount']),
        'distance_from_home': int(request.form['distance_from_home']),
        'transaction_hour': int(request.form['transaction_hour']),
        'merchant_category': request.form['merchant_category'],
        'merchant_type': request.form['merchant_type'],
        'merchant': request.form['merchant'],
        'currency': request.form['currency'],
        'country': request.form['country'],
        'city': request.form['city'],
        'city_size': request.form['city_size'],
        'card_type': request.form['card_type'],
        'device': request.form['device'],
        'channel': request.form['channel'],
        'day_of_week': int(request.form['day_of_week']),
        'is_weekend': int(request.form['is_weekend']),
        'num_transactions_last_hour': int(request.form['num_transactions_last_hour']),
        'total_amount_last_hour': float(request.form['total_amount_last_hour']),
    }
    input_data = pd.DataFrame(data, index=[0])

    # Make prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0]

    # Interpret prediction result
    result = {
        'prediction': 'Fraudulent Transaction' if prediction[0] == 1 else 'Non-Fraudulent Transaction',
        'probability_of_non_fraud': round(probability[0], 2),
        'probability_of_fraud': round(probability[1], 2)
    }

    return jsonify(result)  # Return JSON response

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
