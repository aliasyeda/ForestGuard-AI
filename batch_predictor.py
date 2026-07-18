# batch_predictor.py
"""
Batch Prediction Engine for ForestGuard AI
Process thousands of customers at once
"""

import pandas as pd
import numpy as np
import joblib
import os
import uuid
from datetime import datetime
from typing import Dict, List, Any
import logging
from database import db

logger = logging.getLogger(__name__)

class BatchPredictor:
    """Batch prediction engine"""
    
    def __init__(self, model_path="models/v1/churn_model.pkl", 
                 features_path="models/v1/feature_names.pkl"):
        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(features_path)
    
    def preprocess_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess batch data for prediction"""
        
        # Create feature dictionary
        features_list = []
        
        for idx, row in df.iterrows():
            input_dict = {f: 0 for f in self.feature_names}
            
            # Fill features
            input_dict['tenure'] = row.get('tenure', 0)
            input_dict['MonthlyCharges'] = row.get('monthly_charges', 0)
            input_dict['TotalCharges'] = row.get('total_charges', 0)
            input_dict['SeniorCitizen'] = 1 if row.get('senior_citizen') == "Yes" else 0
            
            # Contract type
            contract = row.get('contract', 'Month-to-month')
            if contract == "Month-to-month":
                input_dict['Contract_Month-to-month'] = 1
            elif contract == "One year":
                input_dict['Contract_One year'] = 1
            else:
                input_dict['Contract_Two year'] = 1
            
            # Engineered features
            tenure = input_dict['tenure']
            monthly_charges = input_dict['MonthlyCharges']
            input_dict['charges_per_tenure'] = monthly_charges / (tenure + 1)
            input_dict['high_risk'] = 1 if (contract == "Month-to-month" and monthly_charges > 70) else 0
            
            features_list.append(input_dict)
        
        return pd.DataFrame(features_list)
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on batch data"""
        
        # Preprocess
        X = self.preprocess_batch(df)
        X = X[self.feature_names]
        
        # Predict
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = (probabilities > 0.5).astype(int)
        
        # Create results dataframe
        results = df.copy()
        results['churn_probability'] = probabilities
        results['churn_prediction'] = predictions
        results['risk_level'] = results['churn_probability'].apply(
            lambda x: 'HIGH' if x > 0.5 else 'MEDIUM' if x > 0.3 else 'LOW'
        )
        
        return results
    
    def save_results(self, results: pd.DataFrame, output_dir: str = "data/predictions"):
        """Save batch prediction results"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        batch_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_predictions_{timestamp}_{batch_id}.csv"
        filepath = os.path.join(output_dir, filename)
        
        results.to_csv(filepath, index=False)
        
        # Save to database
        db.save_batch_prediction({
            'batch_id': batch_id,
            'timestamp': datetime.now().isoformat(),
            'total_customers': len(results),
            'file_name': filename,
            'results_path': filepath
        })
        
        return filepath, batch_id
    
    def get_summary_stats(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics from batch predictions"""
        
        return {
            'total_customers': len(results),
            'churn_rate': results['churn_prediction'].mean(),
            'avg_probability': results['churn_probability'].mean(),
            'high_risk_count': (results['risk_level'] == 'HIGH').sum(),
            'medium_risk_count': (results['risk_level'] == 'MEDIUM').sum(),
            'low_risk_count': (results['risk_level'] == 'LOW').sum(),
            'revenue_at_risk': results['churn_probability'].sum() * 1200  # $1200 avg value
        }

# Example usage
if __name__ == "__main__":
    # Create sample batch
    sample_data = pd.DataFrame({
        'tenure': [12, 24, 6],
        'monthly_charges': [65, 45, 95],
        'total_charges': [780, 1080, 570],
        'senior_citizen': ['No', 'Yes', 'No'],
        'contract': ['Month-to-month', 'One year', 'Month-to-month']
    })
    
    predictor = BatchPredictor()
    results = predictor.predict_batch(sample_data)
    print(results)
    print(predictor.get_summary_stats(results))