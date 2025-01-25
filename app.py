from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = joblib.load('models/logreg_fraud_detection_model.pkl')
scaler = StandardScaler()

@app.route('/')
def home():
    return "Welcome to the Fraud Detection API!"
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()


    try:
       features = np.array(data['features']).reshape(1, -1)
       scaled_features = scaler.fit_transform(features)
       prediction = model.predict(scaled_features)
       probability = model.predict_proba(scaled_features)[:, 1]

       result = {
          'prediction': int(prediction[0]),
          'probability': float(probability[0])
       }

       return jsonify(result)

    except Exception as e:
       return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)