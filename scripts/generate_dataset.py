import numpy as np
import pandas as pd
np.random.seed(42)

n_users = 5000
fraud_rate = 0.05
churn_rate = 0.20

#Labels
fraud_labels = np.random.choice([0, 1], size=n_users, p=[1- fraud_rate, fraud_rate])
churn_labels = np.random.choice([0, 1], size=n_users, p=[1- churn_rate, churn_rate])

#Transactional Features
transaction_amount = np.where(fraud_labels == 1, np.random.uniform(500, 5000, n_users), np.random.uniform(10, 500, n_users))
transaction_hour = np.where(fraud_labels == 1, np.random.randint(0, 6, n_users), np.random.randint(6, 23, n_users))
transaction_frequency = np.where(fraud_labels == 1, np.random.randint(10, 50, n_users), np.random.randint(0, 10, n_users))
unusual_location = np.where(fraud_labels == 1, np.random.choice([0, 1], size=n_users, p=[0.30, 0.70]), np.random.choice([0, 1], size=n_users, p=[0.95, 0.05]))
account_age_days = np.where(fraud_labels == 1, np.random.randint(1, 180, n_users), np.random.randint(180, 3650, n_users))

#Behavioral Features
login_frequency = np.where(churn_labels == 1, np.random.randint(1, 10, n_users), np.random.randint(10, 50, n_users))
support_contacts = np.where(churn_labels == 1, np.random.randint(3, 10, n_users), np.random.randint(0, 2, n_users))
notifications_opted_out = np.where(churn_labels == 1, np.random.choice([0, 1], size=n_users, p=[0.20, 0.80]), np.random.choice([0, 1], size=n_users, p=[0.90, 0.10]))
payment_method = np.random.choice(['credit_card', 'mobile_payment', 'bank_transfer'], size=n_users)
user_id = np.array([f"USR_{i:04d}" for i in range(1, n_users + 1)])

overlap_rate = 0.15
n_overlap = int(n_users * overlap_rate)
overlap_idx = np.random.choice(n_users, size=n_overlap, replace=False)

transaction_amount[overlap_idx] = np.where(
    fraud_labels[overlap_idx] == 1,
    np.random.uniform(10, 500, n_overlap),
    np.random.uniform(500, 5000, n_overlap)
)
transaction_hour[overlap_idx] = np.where(
    fraud_labels[overlap_idx] == 1,
    np.random.randint(6, 23, n_overlap),
    np.random.randint(0, 6, n_overlap)
)
transaction_frequency[overlap_idx] = np.where(
    fraud_labels[overlap_idx] == 1,
    np.random.randint(0, 10, n_overlap),
    np.random.randint(10, 50, n_overlap)
)
login_frequency[overlap_idx] = np.where(
    churn_labels[overlap_idx] == 1,
    np.random.randint(10, 50, n_overlap),
    np.random.randint(1, 10, n_overlap)
)
support_contacts[overlap_idx] = np.where(
    churn_labels[overlap_idx] == 1,
    np.random.randint(0, 2, n_overlap),
    np.random.randint(3, 10, n_overlap)
)

#Dataset Creation
df = pd.DataFrame({
    'user_id': user_id,
    'transaction_amount': transaction_amount,
    'transaction_hour': transaction_hour,
    'transaction_frequency': transaction_frequency,
    'unusual_location': unusual_location,
    'account_age_days': account_age_days,
    'login_frequency': login_frequency,
    'support_contacts': support_contacts,
    'notifications_opted_out': notifications_opted_out,
    'payment_method': payment_method,
    'fraud_risk': fraud_labels,
    'churn_risk': churn_labels
})
df.to_csv("data/raw/synthetic_user_data.csv", index=False)
