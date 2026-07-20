# create_initial_data.py
"""
Create initial data for ForestGuard AI
Sets up database, creates default users, and initial model version
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

print("="*60)
print("ForestGuard AI - Initial Setup")
print("="*60)

# Create directories
dirs = [
    'data/raw',
    'data/processed',
    'data/predictions',
    'models/v1',
    'database',
    'reports',
    'reports/shap',
    'reports/monitoring'
]

for dir_path in dirs:
    os.makedirs(dir_path, exist_ok=True)
    print(f"✅ Created: {dir_path}")

# Load and process data
print("\n📊 Loading dataset...")

# Try to load from different locations
dataset_path = None
for path in ['Telco_Customer_Churn_Dataset .csv', 'data/raw/Telco_Customer_Churn_Dataset .csv']:
    if os.path.exists(path):
        dataset_path = path
        break

if dataset_path is None:
    print("❌ Dataset not found! Please download Telco Customer Churn dataset")
    sys.exit(1)

df = pd.read_csv(dataset_path)
print(f"✅ Dataset loaded: {df.shape}")

# Preprocess
print("\n🔧 Preprocessing data...")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

y = df['Churn'].map({'Yes': 1, 'No': 0})
X = df.drop('Churn', axis=1)
X_encoded = pd.get_dummies(X, drop_first=True)

# Feature engineering
if 'tenure' in X_encoded.columns:
    X_encoded['tenure_group'] = pd.cut(X_encoded['tenure'], 
                                       bins=[0, 12, 24, 48, 72], 
                                       labels=['new', 'moderate', 'loyal', 'veteran'])
    tenure_dummies = pd.get_dummies(X_encoded['tenure_group'], prefix='tenure')
    X_encoded = pd.concat([X_encoded, tenure_dummies], axis=1)
    X_encoded = X_encoded.drop('tenure_group', axis=1)

if 'MonthlyCharges' in X_encoded.columns and 'tenure' in X_encoded.columns:
    X_encoded['charges_per_tenure'] = X_encoded['MonthlyCharges'] / (X_encoded['tenure'] + 1)

if 'Contract_Month-to-month' in X_encoded.columns and 'MonthlyCharges' in X_encoded.columns:
    X_encoded['high_risk'] = ((X_encoded['Contract_Month-to-month'] == 1) & 
                              (X_encoded['MonthlyCharges'] > 70)).astype(int)

print(f"✅ Features: {X_encoded.shape[1]}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE for imbalance
print("\n⚖️ Handling class imbalance...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"Before SMOTE: {X_train.shape} -> After: {X_train_resampled.shape}")

# Train model
print("\n🌲 Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_resampled, y_train_resampled)

# Evaluate
print("\n📊 Evaluating model...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Save model and artifacts
print("\n💾 Saving model and artifacts...")

# Save to v1
model_path = 'models/v1/churn_model.pkl'
feature_names_path = 'models/v1/feature_names.pkl'
metadata_path = 'models/v1/metadata.pkl'

joblib.dump(model, model_path)
joblib.dump(X_encoded.columns.tolist(), feature_names_path)

metadata = {
    'model_type': 'RandomForestClassifier',
    'version': 'v1',
    'created_at': datetime.now().isoformat(),
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc,
    'features_count': X_encoded.shape[1],
    'training_samples': len(X_train_resampled),
    'n_estimators': 200,
    'max_depth': 15
}
joblib.dump(metadata, metadata_path)

print(f"✅ Model saved: {model_path}")
print(f"✅ Features saved: {feature_names_path}")
print(f"✅ Metadata saved: {metadata_path}")

# Save processed data
X_encoded.to_csv('data/processed/X_processed.csv', index=False)
pd.DataFrame(y, columns=['Churn']).to_csv('data/processed/y_processed.csv', index=False)
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

print("✅ Processed data saved")

# Create users
print("\n👤 Creating users...")

users = {
    "admin": {
        "username": "admin",
        "password_hash": "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918",  # admin123
        "role": "admin",
        "created_at": datetime.now().isoformat(),
        "is_active": True
    },
    "analyst": {
        "username": "analyst",
        "password_hash": "a11e5c9d0d6c2a54b2b54b4f8b5e1a8e9e8b7c6d5e4f3a2b1c0d9e8f7a6b5c4",  # analyst123
        "role": "analyst",
        "created_at": datetime.now().isoformat(),
        "is_active": True
    }
}

with open('data/users.json', 'w') as f:
    json.dump(users, f, indent=2)

print("✅ Users created: admin, analyst")

print("\n" + "="*60)
print("🎉 INITIAL SETUP COMPLETE!")
print("="*60)
print("\n📋 Next Steps:")
print("1. Start the API: uvicorn api:app --reload")
print("2. Start the UI: streamlit run app.py")
print("3. Login with: admin / admin123")
print("\n🔗 Access:")
print("   API Docs: http://localhost:8000/docs")
print("   UI: http://localhost:8501")
print("   Monitoring: http://localhost:3000 (Grafana)")