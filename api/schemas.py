from pydantic import BaseModel
class UserFeatures(BaseModel):
    transaction_amount: float
    transaction_hour: int
    transaction_frequency: int
    unusual_location: bool
    account_age_days: int
    login_frequency: int
    notifications_opted_out: bool
    support_contacts: int
    payment_method: str

class RiskPrediction(BaseModel):
    fraud_risk: float
    churn_risk: float