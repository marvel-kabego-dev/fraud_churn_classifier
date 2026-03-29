import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter
from api.schemas import UserFeatures, RiskPrediction


model = joblib.load('../models/model_rf.joblib')
scaler = joblib.load('../models/scaler.joblib')

router = APIRouter()

cols_to_scale = ['transaction_amount', 'transaction_hour', 
                 'transaction_frequency', 'account_age_days',
                 'login_frequency', 'support_contacts']

@router.post('/predict', response_model=RiskPrediction)
def predict(user: UserFeatures):
    df = pd.DataFrame([user.model_dump()])
    df = pd.get_dummies(df, columns=['payment_method'], drop_first=True)
    expected_cols = ['payment_method_mobile_payment', 'payment_method_bank_transfer']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    predict_probas = model.predict_proba(df)
    fraud_score = predict_probas[0][0][1]
    churn_score = predict_probas[1][0][1]

    return RiskPrediction(fraud_risk=fraud_score, churn_risk=churn_score)