# retrain_model.py
"""
Retrain the model with current package versions
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🔄 RETRAINING MODEL WITH CURRENT VERSIONS")
print("=" * 80)

# Create directories
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Load preprocessed data
print("\n📂 Loading preprocessed data...")

try:
    # Try to load existing processed data
    X = pd.read_csv('data/processed/X_processed.csv')
    y = pd.read_csv('data/processed/y_processed.csv')['Churn']
    print(f"✅ Loaded existing processed data: {X.shape}")
except:
    print("⚠️ Processed data not found. Creating from raw data...")
    
    # Load raw data
    try:
        df = pd.read_csv('data/raw/Telco_Customer_Churn_Dataset.csv')
    except:
        df = pd.read_csv('Telco_Customer_Churn_Dataset.csv')
    
    print(f"✅ Raw data loaded: {df.shape}")
    
    # Clean data
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Encode target
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # One-hot encode features
    X = df.drop('Churn', axis=1)
    X = pd.get_dummies(X, drop_first=True)
    
    # Save processed data
    X.to_csv('data/processed/X_processed.csv', index=False)
    pd.DataFrame(y, columns=['Churn']).to_csv('data/processed/y_processed.csv', index=False)
    print(f"✅ Processed data saved: {X.shape}")

# Split data
print("\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training: {X_train.shape}")
print(f"Testing: {X_test.shape}")

# Handle class imbalance with SMOTE if available
try:
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("✅ Using SMOTE for class imbalance")
    print(f"After SMOTE: {X_train_resampled.shape}")
except:
    X_train_resampled, y_train_resampled = X_train, y_train
    print("⚠️ SMOTE not available, using original data")

# Train Random Forest model
print("\n🌲 Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_resampled, y_train_resampled)
print("✅ Model training complete!")

# Evaluate model
print("\n📊 Evaluating model...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")
print("=" * 60)

# Save model and artifacts
print("\n💾 Saving model and artifacts...")

joblib.dump(model, 'models/churn_model.pkl')
joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')

# Save metadata
metadata = {
    'model_type': 'RandomForestClassifier',
    'n_estimators': 200,
    'max_depth': 15,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc,
    'features_count': X.shape[1],
    'training_samples': len(X_train_resampled)
}
joblib.dump(metadata, 'models/model_metadata.pkl')

print("✅ Model saved: models/churn_model.pkl")
print("✅ Feature names saved: models/feature_names.pkl")
print("✅ Metadata saved: models/model_metadata.pkl")

# Save split data for future use
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)
print("✅ Split data saved to data/processed/")

# Feature importance visualization
print("\n📊 Creating feature importance visualization...")
import matplotlib.pyplot as plt
import seaborn as sns

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
plt.title('Top 15 Features Driving Customer Churn', fontsize=16, pad=20)
plt.xlabel('Feature Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.savefig('reports/feature_importance_top15.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: reports/feature_importance_top15.png")

# Save test predictions for evaluation
print("\n💾 Saving test predictions...")
pd.DataFrame({'actual': y_test, 'predicted': y_pred, 'probability': y_proba}).to_csv('data/processed/test_predictions.csv', index=False)
print("✅ Test predictions saved: data/processed/test_predictions.csv")

print("\n" + "=" * 80)
print("✅ MODEL RETRAINING COMPLETE!")
print("=" * 80)
print("\n🚀 Now you can run:")
print("   streamlit run app.py")
print("   python run_shap_analysis.py")