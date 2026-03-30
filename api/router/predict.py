import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter
from api.schemas import UserFeatures, RiskPrediction


model = joblib.load('models/model_rf.joblib')
scaler = joblib.load('models/scaler.joblib')

router = APIRouter()

cols_to_scale = ['transaction_amount', 'transaction_hour', 
                 'transaction_frequency', 'account_age_days',
                 'login_frequency', 'support_contacts']

@router.post('/predict', response_model=RiskPrediction)
def predict(user: UserFeatures):
    df = pd.DataFrame([user.model_dump()])
    payment = df['payment_method'].iloc[0]
    df['payment_method_credit_card'] = 1 if payment == 'credit_card' else 0
    df['payment_method_mobile_payment'] = 1 if payment == 'mobile_payment' else 0
    df = df.drop(columns=['payment_method'])
    feature_order = [
    'transaction_amount', 'transaction_hour', 'transaction_frequency',
    'unusual_location', 'account_age_days', 'login_frequency',
    'support_contacts', 'notifications_opted_out',
    'payment_method_credit_card', 'payment_method_mobile_payment'
    ]
    df = df[feature_order]
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    predict_probas = model.predict_proba(df)
    fraud_score = predict_probas[0][0][1]
    churn_score = predict_probas[1][0][1]

    return RiskPrediction(fraud_risk=fraud_score, churn_risk=churn_score)