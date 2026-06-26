import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Regenerating 5,000 Rows with New Balanced Names
num_rows = 5000
np.random.seed(42)

ages = np.random.randint(18, 31, size=num_rows)
genders = np.random.choice(['Male', 'Female', 'Other'], size=num_rows, p=[0.49, 0.49, 0.02])
qualifications = np.random.choice(['High School', 'Diploma', 'Undergraduate', 'Postgraduate'], 
                                  size=num_rows, p=[0.15, 0.20, 0.50, 0.15])

financial_goals = []
for age in ages:
    if age <= 22:
        goal = np.random.choice(['Income Generation', 'Wealth Optimization'], p=[0.75, 0.25])
    else:
        goal = np.random.choice(['Income Generation', 'Wealth Optimization'], p=[0.40, 0.60])
    financial_goals.append(goal)

df = pd.DataFrame({
    'Age': ages,
    'Gender': genders,
    'Qualification': qualifications,
    'Financial_Goal': financial_goals
})

target_paths = []
for i in range(num_rows):
    goal = df['Financial_Goal'].iloc[i]
    qual = df['Qualification'].iloc[i]
    age = df['Age'].iloc[i]
    
    # 2-Earning & 2-Wealth Balanced Categories
    if goal == 'Income Generation':
        if qual in ['Undergraduate', 'Postgraduate']:
            p_dist = [0.70, 0.15, 0.10, 0.05]
        else:
            p_dist = [0.15, 0.65, 0.15, 0.05]
    elif goal == 'Wealth Optimization':
        if age >= 25:
            p_dist = [0.05, 0.05, 0.20, 0.70]
        else:
            p_dist = [0.10, 0.10, 0.60, 0.20]

    chosen_path = np.random.choice(
        ['Tech Side-Hustler', 'Micro-Earning Starter', 'Tactical Growth Investor', 'Defensive Capital Saver'], 
        p=p_dist
    )
    target_paths.append(chosen_path)

df['Financial_Path'] = target_paths
df.to_csv('financity_ux_optimized_dataset.csv', index=False)

# 2. Encoding and Re-Training Model
X = df[['Age', 'Gender', 'Qualification', 'Financial_Goal']].copy()
y = df['Financial_Path'].copy()

feature_encoders = {}
for col in ['Gender', 'Qualification', 'Financial_Goal']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    feature_encoders[col] = le

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

final_rf = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=5, random_state=42)
final_rf.fit(X_train, y_train)

print(f"🎯 Model Re-Trained Successfully! Restored Accuracy: {accuracy_score(y_test, final_rf.predict(X_test)) * 100:.2f}%")

# 3. Serialization
os.makedirs('models', exist_ok=True)
joblib.dump(final_rf, 'models/rf_category_predictor.pkl')
joblib.dump(feature_encoders, 'models/feature_encoders.pkl')
joblib.dump(target_encoder, 'models/target_encoder.pkl')
print("📦 All balanced model components dumped into 'models/' folder!")