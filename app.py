from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import xgboost as xgb
import json
from datetime import datetime
import requests

WEEK_DATE_RANGES = {
    "week1": ("2024-09-05", "2024-09-11"),
    "week2": ("2024-09-12", "2024-09-18"),
    "week3": ("2024-09-19", "2024-09-25"),
    "week4": ("2024-09-26", "2024-10-02"),
    "week5": ("2024-10-03", "2024-10-09"),
    "week6": ("2024-10-10", "2024-10-16"),
    "week7": ("2024-10-17", "2024-10-23"),
    "week8": ("2024-10-24", "2024-10-30"),
    "week9": ("2024-10-31", "2024-11-06"),
    "week10": ("2024-11-07", "2024-11-13"),
    "week11": ("2024-11-14", "2024-11-20"),
    "week12": ("2024-11-21", "2024-11-27"),
}

def parse_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d")

def get_players_for_week(week):
    start_str, end_str = WEEK_DATE_RANGES.get(week, (None, None))
    if not start_str or not end_str:
        return []

    start_date = parse_date(start_str)
    end_date = parse_date(end_str)

    with open('static/skill_players.json') as f:
        skill_players = json.load(f)

    with open('static/qb_players.json') as f:
        qb_players = json.load(f)

    with open('static/player_performances.json') as f:
        skill_perfs = json.load(f)

    with open('static/qb_performances.json') as f:
        qb_perfs = json.load(f)

    player_inputs = []

    for player in skill_players:
        name = player.lower()

        perf = next(
            (p for p in skill_perfs if p['Player'].lower() == name and start_date <= parse_date(p['Date']) <= end_date),
            None
        )

        if perf:
            features = [
                perf['Receptions'],
                perf['Receiving Yards'],
                perf['Receiving TD'],
                perf['Rushing Yards'],
                perf['Rushing TD'],
                perf['Fumbles Lost']
            ]
            player_inputs.append({
                'name': name,
                'position': 'Other',
                'features': features
            })


    for player in qb_players:
        name = player.lower()
        perf = next(
            (
                p for p in qb_perfs
                if p['Player'].lower() == name and start_date <= parse_date(p['Date']) <= end_date
            ),
            None
        )
        if perf:
            features = [
                perf['Passing Yards'],
                perf['Passing TD'],
                perf['Passing INT'],
                perf['Rushing Yards'],
                perf['Rushing TD'],
                perf['Fumbles Lost']
            ]
            player_inputs.append({
                'name': name,
                'position': 'QB',
                'features': features
            })

    return player_inputs

app = Flask(__name__)
CORS(app)

model = xgb.XGBRegressor()
model.load_model('CombinedModels/offensive_players_model.json')

qb_model = xgb.XGBRegressor()
qb_model.load_model('QBs/QB_model.json')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    log_prediction = model.predict(features)
    prediction = np.expm1(log_prediction)
    return jsonify({'prediction': prediction.tolist()})

@app.route('/predict-qb', methods=['POST'])
def predict_qb_points():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = qb_model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

@app.route('/qb-prediction')
def predict_qb():
    return render_template('predict_qb.html')

@app.route('/skill-prediction')
def predict_skill():
    return render_template('predict_skill.html')

@app.route('/recommend')
def recommend():
    return render_template('recommend.html')

@app.route('/get-week-recommendations')
def get_week_recommendations():
    week = request.args.get('week')
    players = get_players_for_week(week)

    results = []
    print('Starting...')
    for p in players:
        features = np.array(p['features']).reshape(1, -1)
        if p['position'] == 'QB':
            pred = qb_model.predict(features)[0]
        else:
            log_pred = model.predict(features)[0]
            pred = np.expm1(log_pred)

        results.append({
            'name': p['name'],
            'points': float(pred)
        })

    results.sort(key=lambda x: x['points'], reverse=True)
    return jsonify({'predictions': results[:20]})

if __name__ == '__main__':
    app.run(debug=True)
