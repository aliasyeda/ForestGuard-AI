# test_model_load.py
import joblib
import os

print("Checking model files...")

model_path = 'models/churn_model.pkl'
if os.path.exists(model_path):
    print(f"✅ Model found: {model_path}")
    model = joblib.load(model_path)
    print(f"   Model type: {type(model).__name__}")
    print(f"   Model parameters: {model.get_params()}")
else:
    print(f"❌ Model not found at: {model_path}")

# Check feature names
feature_path = 'models/feature_names.pkl'
if os.path.exists(feature_path):
    features = joblib.load(feature_path)
    print(f"✅ Feature names loaded: {len(features)} features")
    print(f"   First 5 features: {features[:5]}")
else:
    print(f"❌ Feature names not found")

print("\n✅ Test complete!")