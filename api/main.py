from fastapi import FastAPI
from api.router import predict

app = FastAPI(title="Fraud & Churn Risk Classifier")

app.include_router(predict.router)