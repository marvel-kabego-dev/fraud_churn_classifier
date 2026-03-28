import numpy as np
import pandas as pd
n_users = 5000
fraud_rate = 0.05
churn_rate = 0.20
fraud_labels = np.random.choice([0, 1], size=n_users, p=[1- fraud_rate, fraud_rate])
churn_labels = np.random.choice([0, 1], size=n_users, p=[1- churn_rate, churn_rate])
transaction_amount = np.where(fraud_labels == 1, np.random.uniform(500, 5000, n_users), np.random.uniform(10, 500, n_users))
transaction_hour = np.where(fraud_labels == 1, np.random.randint(0, 6, n_users), np.random.randint(6, 23, n_users))
transaction_frequency = np.where(fraud_labels == 1, np.random.randint(10, 50, n_users), np.random.randint(0, 10, n_users))
unusual_location = np.where(fraud_labels == 1, np.random.choice([0, 1], size=n_users, p=[0.30, 0.70]), np.random.choice([0, 1], size=n_users, p=[0.95, 0.05]))
account_age_days = np.where(fraud_labels == 1, np.random.randint(1, 180, n_users), np.random.randint(180, 3650, n_users))
login_frequency = np.where(churn_labels == 1, np.random.randint(1, 10, n_users), np.random.randint(10, 50, n_users))


