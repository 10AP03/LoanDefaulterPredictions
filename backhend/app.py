from flask import Flask,request,jsonify

app = Flask(__name__)

import pickle 
import numpy as np

with open("loan_defaulter_model.pkl","rb") as f:
    model = pickle.load(f)

with open("scaler.pkl","rb") as f:
  scaler = pickle.load(f)

frozen_feature_list = [

      "Age",
      "Income",
      "LoanAmount",
      "CreditScore",
      "MonthsEmployed",
      "NumCreditLines",
      "InterestRate",
      "LoanTerm",
      "DTIRatio",
      "HasMortgage",
      "HasDependents"
    ]


@app.route('/')
def index():
  return "API is Running"

@app.route('/predict', methods = ['POST'])
def predict():
  
    data = request.get_json()


    if data is None:
      return jsonify({
      "status": "error",
      "message": "Missing required fields"
    }), 400

    
    missing_fields = []

    for field in frozen_feature_list:
      if field not in data:
        missing_fields.append(field)

    
    if missing_fields:
      return jsonify({
      "status": "error",
      "message": "Missing required fields",
      "missing_fields": missing_fields
    }), 400

    
    expected_fields = {

    "Age":int,
    "Income":int,
    "LoanAmount":int,
    "CreditScore":int,
    "MonthsEmployed":int,
    "NumCreditLines":int,
    "InterestRate":(int,float),
    "LoanTerm":int,
    "DTIRatio":(int,float),
    "HasMortgage":int,
    "HasDependents":int
    }
    
    type_errors = []

    for field, expected_type in expected_fields.items():
      if not isinstance(data[field], expected_type):
        type_errors.append(field)

    
    if type_errors:
      return jsonify({
      "status": "error",
      "message": "Invalid data types",
      "fields": type_errors
    }), 400

    value_rules = {
    "Age": (0, 120),
    "Income": (0, None),
    "LoanAmount": (0, None),
    "CreditScore": (300, 850),
    "MonthsEmployed": (0, None),
    "NumCreditLines": (0, None),
    "InterestRate": (0, 100),
    "LoanTerm": (1, None),
    "DTIRatio": (0, 1),
    "HasMortgage": (0, 1),
    "HasDependents": (0, None)
  }
    value_errors = []

    for field, (min_val, max_val) in value_rules.items():
      value = data[field]

      if min_val is not None and value < min_val:
        value_errors.append(field)

      if max_val is not None and value > max_val:
        value_errors.append(field)


    if value_errors:
      return jsonify({
      "status": "error",
      "message": "Invalid field values",
      "fields": value_errors
    }), 400

    model_input = []

    for feature in frozen_feature_list:
      model_input.append(data[feature])

    model_input_array = np.array(model_input).reshape(1, -1)
    scaled_input = scaler.transform(model_input_array)


    prediction = model.predict(scaled_input)
    prediction_value = int(prediction[0])


    probability = model.predict_proba(scaled_input)[0][1]

    threshold = 0.4

    loan_approved = 0 if probability < threshold else 1

    if probability < 0.30:
      risk_level = "LOW"

    elif 0.30 <= probability <= 0.60:
      risk_level = "MEDIUM"

    elif probability > 0.60:
      risk_level = "HIGH"
 
    return jsonify({
        "status": "success",
        "prediction": prediction_value,
        "default probability": round(float(probability),4),
        "risk_level":risk_level,
        "threshold":threshold
    })
    


if __name__=='__main__':          #running the app
  app.run(host ='0.0.0.0',port = 5000)   #debug mode is on for development purposes