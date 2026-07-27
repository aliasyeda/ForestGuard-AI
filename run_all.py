# run_all.py
"""
Complete Pipeline - Run all steps in order
This script will execute all preprocessing, training, and evaluation steps
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 CUSTOMER CHURN PREDICTION - COMPLETE PIPELINE")
print("="*80)

# Create necessary directories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

print("\n✅ Directories created")

# Step 1: Check if dataset exists
dataset_path = None
possible_paths = [
    'data/raw/Telco_Customer_Churn_Dataset .csv',
    'Telco_Customer_Churn_Dataset .csv',
    '../Telco_Customer_Churn_Dataset .csv'
]

for path in possible_paths:
    if os.path.exists(path):
        dataset_path = path
        break

if dataset_path is None:
    print("\n❌ ERROR: Dataset not found!")
    print("Please place Telco_Customer_Churn_Dataset.csv in the 'data/raw/' folder")
    exit()

print(f"\n✅ Dataset found at: {dataset_path}")

# Step 2: Load and preprocess data
print("\n" + "="*80)
print("STEP 1: DATA PREPROCESSING")
print("="*80)

df = pd.read_csv(dataset_path)
print(f"✅ Data loaded: {df.shape}")

# Clean data
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

# Encode target
y = df['Churn'].map({'Yes': 1, 'No': 0})

# One-hot encode features
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

# Save preprocessed data
X_encoded.to_csv('data/processed/X_processed.csv', index=False)
pd.DataFrame(y, columns=['Churn']).to_csv('data/processed/y_processed.csv', index=False)

print(f"✅ Preprocessed data saved")
print(f"   Features: {X_encoded.shape}")
print(f"   Target distribution: {y.value_counts().to_dict()}")

# Step 3: Split data
print("\n" + "="*80)
print("STEP 2: DATA SPLITTING")
print("="*80)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Save split data
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

print(f"✅ Data split completed")
print(f"   Training: {len(X_train)} samples")
print(f"   Testing: {len(X_test)} samples")

# Step 4: Train model
print("\n" + "="*80)
print("STEP 3: MODEL TRAINING")
print("="*80)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

best_auc = 0
best_model = None
best_name = ""

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    
    if roc_auc > best_auc:
        best_auc = roc_auc
        best_model = model
        best_name = name

print(f"\n🏆 Best Model: {best_name} (ROC-AUC: {best_auc:.4f})")

# Save model and artifacts
import joblib

joblib.dump(best_model, 'models/churn_model.pkl')
joblib.dump(X_train.columns.tolist(), 'models/feature_names.pkl')

model_metadata = {
    'best_model_name': best_name,
    'best_roc_auc': best_auc,
    'features_count': X_train.shape[1],
    'training_samples': len(X_train_resampled)
}
joblib.dump(model_metadata, 'models/model_metadata.pkl')

print(f"\n✅ Model saved to: models/churn_model.pkl")
print(f"✅ Feature names saved to: models/feature_names.pkl")
print(f"✅ Metadata saved to: models/model_metadata.pkl")

# Step 5: Evaluate model
print("\n" + "="*80)
print("STEP 4: MODEL EVALUATION")
print("="*80)

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"\n📊 Final Model Performance:")
print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"   F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"   ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")

# Save evaluation results
evaluation_results = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc
}
joblib.dump(evaluation_results, 'models/evaluation_results.pkl')

print("\n" + "="*80)
print("✅ COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
print("="*80)
print("\n📁 Files Created:")
print("   📂 data/processed/")
print("      - X_processed.csv")
print("      - y_processed.csv")
print("      - X_train.csv, X_test.csv")
print("      - y_train.csv, y_test.csv")
print("   📂 models/")
print("      - churn_model.pkl")
print("      - feature_names.pkl")
print("      - model_metadata.pkl")
print("      - evaluation_results.pkl")
print("\n🚀 Now you can run: streamlit run app.py")