import numpy as np
import pandas as pd
n_users = 5000
fraud_rate = 0.05
churn_rate = 0.20
fraud_labels = np.random.choice([0, 1], size=n_users, p=[1- fraud_rate, fraud_rate])
churn_labels = np.random.choice([0, 1], size=n_users, p=[1- churn_rate, churn_rate])


