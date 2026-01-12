## Loan Default Risk Prediction API
A machine-learning powered REST API that predicts loan default risk based on applicant financial data. 
The API returns default probability and risk classification (LOW / MEDIUM / HIGH).





## Live API
https://loan-defaulter-api.onrender.com





## Tech Stack
- Python
- Flask
- scikit-learn
- NumPy
- Gunicorn
- Render (Cloud Deployment)





## API Endpoints

Health Check
GET /

Response:
API is Running

Predict Loan Default
POST /predict





## Sample Request

{
  "Age": 40,
  "Income": 60000,
  "LoanAmount": 250000,
  "CreditScore": 720,
  "MonthsEmployed": 24,
  "NumCreditLines": 5,
  "InterestRate": 8.5,
  "LoanTerm": 36,
  "DTIRatio": 0.32,
  "HasMortgage": 1,
  "HasDependents": 2
}





## Sample Response

{
  "status": "success",
  "prediction": 0,
  "default_probability": 0.134,
  "risk_level": "LOW",
  "threshold": 0.4
}





  ## Key Features
- Probability-based loan default prediction
- Risk classification (LOW / MEDIUM / HIGH)
- Strict input validation
- Production-ready ML inference
- Deployed cloud API
