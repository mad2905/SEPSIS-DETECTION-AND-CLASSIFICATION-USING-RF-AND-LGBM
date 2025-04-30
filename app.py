from flask import Flask, request, render_template, redirect, url_for, json
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained stacking model
stacking_model = joblib.load('stacking_model_q.joblib')
print("Model loaded successfully")

def classify_sepsis_sirs(row):
    sirs_criteria = 0
    sepsis = False
    severe_sepsis = False
    critical_sepsis = False

    # Check SIRS criteria
    if row.get('Temp', -1) > 38 or row.get('Temp', -1) < 36:
        sirs_criteria += 1
    if row.get('HR', -1) >= 90:
        sirs_criteria += 1
    if row.get('Resp', -1) >= 20:
        sirs_criteria += 1
    if row.get('WBC', -1) > 12000 or row.get('WBC', -1) < 4000:
        sirs_criteria += 1

    # Determine sepsis levels
    if sirs_criteria >= 2:
        sepsis = True

    if sepsis:
        if (
            row.get('SBP', -1) < 90 or row.get('MAP', -1) < 65 or
            row.get('O2Sat', -1) < 90 or row.get('Lactate', -1) > 2 or
            row.get('Creatinine', -1) > 1.5 or row.get('BUN', -1) > 20 or
            row.get('Bilirubin_total', -1) > 2
        ):
            severe_sepsis = True

    if severe_sepsis:
        if (row.get('SBP', -1) < 90 or row.get('MAP', -1) < 65) and (row.get('Lactate', -1) >= 4):
            critical_sepsis = True

    if critical_sepsis:
        return 'Critical Sepsis (Septic Shock)'
    elif severe_sepsis:
        return 'Severe Sepsis'
    elif sepsis:
        return 'SIRS/Sepsis'
    else:
        return 'No Sepsis'

def predict_sepsis(patient_data):
    # Convert to DataFrame
    df = pd.DataFrame([patient_data])
    
    # Ensure column names match training data
    expected_cols = stacking_model.feature_names_in_
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan
    
    # Reorder columns to match training data
    df = df[expected_cols]
    
    # Impute missing values with column means
    df.fillna(df.mean(), inplace=True)
    
    # Get prediction probabilities
    proba = stacking_model.predict_proba(df)[:, 1][0]
    
    # Only perform detailed classification if model predicts sepsis
    if proba > 0.5:
        classification = classify_sepsis_sirs(patient_data)
    else:
        classification = 'No Sepsis'
    
    return {
        "Probability Score": float(proba),
        "Classification": classification,
        "AUC-ROC": 0.92  # From training results
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Map form inputs to model features
            patient_data = {
                'HR': float(request.form['heart_rate']),
                'O2Sat': float(request.form['o2sat']),
                'Temp': float(request.form['temp']),
                'SBP': float(request.form['sbp']),
                'MAP': float(request.form['map']),
                'DBP': float(request.form['dbp']),
                'Resp': float(request.form['resp']),
                'EtCO2': float(request.form['etco2']) if request.form['etco2'] else np.nan,
                'BaseExcess': float(request.form['base_excess']) if request.form['base_excess'] else np.nan,
                'HCO3': float(request.form['hco3']),
                'FiO2': float(request.form['fio2']) if request.form['fio2'] else np.nan,
                'pH': float(request.form['ph']),
                'PaCO2': float(request.form['paco2']),
                'SaO2': float(request.form['sao2']),
                'AST': float(request.form['ast']) if request.form['ast'] else np.nan,
                'BUN': float(request.form['bun']),
                'Alkalinephos': float(request.form['alk_phos']) if request.form['alk_phos'] else np.nan,
                'Calcium': float(request.form['calcium']),
                'Chloride': float(request.form['chloride']),
                'Creatinine': float(request.form['creatinine']),
                'Bilirubin_direct': float(request.form['direct_bilirubin']) if request.form['direct_bilirubin'] else np.nan,
                'Glucose': float(request.form['glucose']),
                'Lactate': float(request.form['lactate']),
                'Magnesium': float(request.form['magnesium']),
                'Phosphate': float(request.form['phosphate']),
                'Potassium': float(request.form['potassium']),
                'Bilirubin_total': float(request.form['total_bilirubin']),
                'TroponinI': float(request.form['troponin']) if request.form['troponin'] else np.nan,
                'Hct': float(request.form['hct']),
                'Hgb': float(request.form['hgb']),
                'PTT': float(request.form['ptt']) if request.form['ptt'] else np.nan,
                'WBC': float(request.form['wbc'])
            }

            result = predict_sepsis(patient_data)
            return redirect(url_for('result', result=json.dumps(result)))

        except Exception as e:
            return f"Error processing request: {str(e)}", 400

    return render_template('index1.html')

@app.route('/result')
def result():
    result = json.loads(request.args.get('result'))
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)