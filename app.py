from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import xgboost as xgb

app = Flask(__name__)
CORS(app)

model = xgb.XGBRegressor()
model.load_model('CombinedModels/offensive_players_model.json')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    log_prediction = model.predict(features)
    prediction = np.expm1(log_prediction)
    return jsonify({'prediction': prediction.tolist()})


if __name__ == '__main__':
    app.run(debug=True)
