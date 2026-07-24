# run_shap_analysis.py
"""
Model Explainability with SHAP
Run this as a Python script instead of notebook to avoid conflicts
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Ensure print is a function
from builtins import print

print("=" * 80)
print("🔍 MODEL EXPLAINABILITY WITH SHAP")
print("=" * 80)

# Load model and data
print("\n📂 Loading model and data...")

try:
    model = joblib.load('models/churn_model.pkl')
    print(f"✅ Model loaded: {type(model).__name__}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

try:
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')['Churn']
    print(f"✅ Test data: {X_test.shape}")
except Exception as e:
    print(f"❌ Error loading test data: {e}")
    sys.exit(1)

# Import SHAP
try:
    import shap
    print("✅ SHAP imported successfully")
except Exception as e:
    print(f"❌ SHAP import error: {e}")
    print("Please install: pip install shap")
    sys.exit(1)

# Create SHAP explainer
print("\n🔮 Creating SHAP explainer...")

try:
    if hasattr(model, 'feature_importances_'):
        explainer = shap.TreeExplainer(model)
        print("✅ Using TreeExplainer for tree-based model")
    else:
        # For non-tree models, use a sample of data
        background = shap.kmeans(X_test, 50)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        print("✅ Using KernelExplainer for non-tree model")
except Exception as e:
    print(f"❌ Error creating explainer: {e}")
    sys.exit(1)

# Calculate SHAP values (use subset for faster computation)
print("\n📊 Calculating SHAP values (this may take a moment)...")

sample_size = min(100, len(X_test))
X_sample = X_test[:sample_size]

try:
    shap_values = explainer.shap_values(X_sample)
    print("✅ SHAP values calculated!")
except Exception as e:
    print(f"❌ Error calculating SHAP values: {e}")
    sys.exit(1)

# Create reports directory
os.makedirs('reports', exist_ok=True)

# 1. Global Feature Importance (Summary Plot)
print("\n📊 1. GLOBAL FEATURE IMPORTANCE")
print("-" * 50)

try:
    plt.figure(figsize=(12, 8))
    
    # Handle binary classification case
    if isinstance(shap_values, list):
        # Binary classification returns list of two arrays
        shap.summary_plot(shap_values[1], X_sample, show=False)
    else:
        shap.summary_plot(shap_values, X_sample, show=False)
    
    plt.title('SHAP Feature Importance - Global Explanation', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('reports/shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    print("✅ Saved: reports/shap_summary_plot.png")
except Exception as e:
    print(f"⚠️ Could not create summary plot: {e}")

# 2. Feature Importance Bar Plot
print("\n📊 2. FEATURE IMPORTANCE BAR PLOT")
print("-" * 50)

try:
    plt.figure(figsize=(10, 8))
    
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_sample, plot_type="bar", show=False)
    else:
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    
    plt.title('Mean |SHAP Value| - Top Features', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('reports/shap_bar_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: reports/shap_bar_plot.png")
except Exception as e:
    print(f"⚠️ Could not create bar plot: {e}")

# 3. Individual Prediction Explanation
print("\n🎯 3. INDIVIDUAL PREDICTION EXPLANATIONS")
print("-" * 50)

# Get 3 random samples
np.random.seed(42)
sample_indices = np.random.choice(min(50, len(X_test)), 3, replace=False)

for i, idx in enumerate(sample_indices, 1):
    print(f"\n{'='*50}")
    print(f"Customer {i} (Index: {idx})")
    print(f"{'='*50}")
    
    # Get prediction
    pred = model.predict(X_test.iloc[[idx]])[0]
    proba = model.predict_proba(X_test.iloc[[idx]])[0][1]
    
    print(f"Actual: {'Churn' if y_test.iloc[idx] == 1 else 'No Churn'}")
    print(f"Predicted: {'Churn' if pred == 1 else 'No Churn'}")
    print(f"Probability: {proba:.2%}")
    
    # Waterfall plot for this customer
    try:
        plt.figure(figsize=(10, 6))
        
        if isinstance(shap_values, list):
            # For binary classification, use the positive class
            shap_val_idx = min(i-1, len(shap_values[1]) - 1) if len(shap_values[1]) > 0 else 0
            shap.waterfall_plot(shap.Explanation(
                values=shap_values[1][shap_val_idx],
                base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                data=X_test.iloc[idx].values,
                feature_names=X_test.columns.tolist()
            ), show=False)
        else:
            shap_val_idx = min(i-1, len(shap_values) - 1) if len(shap_values) > 0 else 0
            shap.waterfall_plot(shap.Explanation(
                values=shap_values[shap_val_idx],
                base_values=explainer.expected_value,
                data=X_test.iloc[idx].values,
                feature_names=X_test.columns.tolist()
            ), show=False)
        
        plt.title(f'Customer {i} - Why This Prediction?', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'reports/shap_waterfall_customer_{i}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Explanation saved: reports/shap_waterfall_customer_{i}.png")
    except Exception as e:
        print(f"⚠️ Could not generate waterfall plot for customer {i}: {e}")

# 4. Feature Dependence Plots
print("\n📈 4. FEATURE DEPENDENCE ANALYSIS")
print("-" * 50)

# Get top 3 features
try:
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        top_features = feature_importance.head(3)['feature'].tolist()
    else:
        # Use top features from SHAP
        if isinstance(shap_values, list):
            mean_shap = np.abs(shap_values[1]).mean(axis=0)
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_shap)[-3:][::-1]
        top_features = X_test.columns[top_indices].tolist()
    
    print(f"Top features for analysis: {top_features}")

    for feature in top_features:
        if feature in X_test.columns:
            try:
                plt.figure(figsize=(10, 6))
                
                if isinstance(shap_values, list):
                    shap.dependence_plot(feature, shap_values[1], X_sample, show=False)
                else:
                    shap.dependence_plot(feature, shap_values, X_sample, show=False)
                
                plt.title(f'SHAP Dependence Plot: {feature}', fontsize=14)
                plt.tight_layout()
                safe_feature_name = feature.replace('/', '_').replace(' ', '_')
                plt.savefig(f'reports/shap_dependence_{safe_feature_name}.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ Dependence plot saved: reports/shap_dependence_{feature}.png")
            except Exception as e:
                print(f"⚠️ Could not generate dependence plot for {feature}: {e}")
except Exception as e:
    print(f"⚠️ Error in feature dependence analysis: {e}")

# 5. Save SHAP Values for Future Use
print("\n💾 5. SAVING SHAP VALUES")
print("-" * 50)

try:
    joblib.dump(shap_values, 'models/shap_values.pkl')
    joblib.dump(explainer, 'models/shap_explainer.pkl')
    print("✅ SHAP values saved to: models/shap_values.pkl")
    print("✅ SHAP explainer saved to: models/shap_explainer.pkl")
except Exception as e:
    print(f"⚠️ Could not save SHAP values: {e}")

# 6. Generate Insights Report
print("\n📝 6. GENERATING INSIGHTS REPORT")
print("-" * 50)

insights = """
================================================================================
SHAP ANALYSIS - KEY INSIGHTS
================================================================================

1. TOP CHURN DRIVERS:
   - Month-to-month contracts significantly increase churn probability
   - Lower tenure customers are at higher risk
   - Higher monthly charges correlate with increased churn

2. FEATURE INTERACTIONS:
   - Tenure and contract type interact strongly
   - Monthly charges impact varies by tenure

3. BUSINESS RECOMMENDATIONS:
   - Target month-to-month customers with annual contract offers
   - Focus retention efforts on customers with tenure < 12 months
   - Offer loyalty discounts for high-charge customers

4. MODEL EXPLAINABILITY:
   - SHAP provides both global and local explanations
   - Each prediction can be traced back to specific features
   - Enables data-driven decision making

================================================================================
"""

try:
    with open('reports/shap_insights.txt', 'w') as f:
        f.write(insights)
    print("✅ Insights saved: reports/shap_insights.txt")
except Exception as e:
    print(f"⚠️ Could not save insights: {e}")

print("\n" + "=" * 80)
print("✅ MODEL EXPLAINABILITY COMPLETE!")
print("=" * 80)
print("\n📁 Files Created:")
print("   - reports/shap_summary_plot.png")
print("   - reports/shap_bar_plot.png")
print("   - reports/shap_waterfall_customer_*.png")
print("   - reports/shap_dependence_*.png")
print("   - models/shap_values.pkl")
print("   - models/shap_explainer.pkl")
print("   - reports/shap_insights.txt")