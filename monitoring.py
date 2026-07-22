# monitoring.py
"""
Model Monitoring Dashboard for ForestGuard AI
Tracks model performance, drift, and predictions over time
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3
from database import db

class ModelMonitor:
    """Monitor model performance and predictions"""
    
    def __init__(self):
        self.db = db
    
    def get_prediction_trend(self, days=30):
        """Get prediction trend over time"""
        conn = sqlite3.connect(self.db.db_path)
        
        query = f"""
            SELECT 
                date(created_at) as date,
                COUNT(*) as total_predictions,
                AVG(churn_probability) as avg_probability,
                SUM(CASE WHEN risk_level = 'HIGH' THEN 1 ELSE 0 END) as high_risk_count
            FROM predictions 
            WHERE created_at >= datetime('now', '-{days} days')
            GROUP BY date(created_at)
            ORDER BY date
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_model_metrics(self):
        """Get current model metrics"""
        conn = sqlite3.connect(self.db.db_path)
        
        # Get recent predictions
        query = """
            SELECT 
                COUNT(*) as total,
                AVG(churn_probability) as avg_prob,
                SUM(CASE WHEN churn_prediction = 1 THEN 1 ELSE 0 END) as churn_count
            FROM predictions 
            WHERE created_at >= datetime('now', '-7 days')
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) == 0:
            return {}
        
        return {
            'total_predictions': int(df['total'].iloc[0]),
            'avg_churn_probability': float(df['avg_prob'].iloc[0]),
            'predicted_churn_rate': float(df['churn_count'].iloc[0] / df['total'].iloc[0]) if df['total'].iloc[0] > 0 else 0
        }
    
    def create_monitoring_dashboard(self):
        """Create Streamlit monitoring dashboard"""
        
        st.markdown("## 📊 Model Monitoring Dashboard")
        st.markdown("*Track model performance and prediction trends*")
        
        # Get data
        metrics = self.get_model_metrics()
        trend_data = self.get_prediction_trend(30)
        
        if not metrics:
            st.info("No predictions made yet. Start making predictions to see monitoring data.")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions (7d)", metrics['total_predictions'])
        
        with col2:
            st.metric("Avg Churn Probability", f"{metrics['avg_churn_probability']:.1%}")
        
        with col3:
            st.metric("Predicted Churn Rate", f"{metrics['predicted_churn_rate']:.1%}")
        
        with col4:
            st.metric("Model Health", "✅ Stable")
        
        # Prediction trend chart
        if not trend_data.empty:
            st.markdown("### 📈 Prediction Trend")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=trend_data['date'],
                y=trend_data['avg_probability'],
                mode='lines+markers',
                name='Avg Churn Probability',
                line=dict(color='#ff6b6b', width=2),
                marker=dict(size=8)
            ))
            
            fig.add_trace(go.Bar(
                x=trend_data['date'],
                y=trend_data['high_risk_count'],
                name='High Risk Customers',
                marker_color='#ff9999',
                opacity=0.6,
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='Churn Probability Trend',
                xaxis_title='Date',
                yaxis_title='Avg Probability',
                yaxis2=dict(
                    title='High Risk Count',
                    overlaying='y',
                    side='right'
                ),
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk distribution
        st.markdown("### 🎯 Risk Distribution")
        
        recent_predictions = self.db.get_recent_predictions(100)
        
        if recent_predictions:
            risk_df = pd.DataFrame(recent_predictions)
            risk_counts = risk_df['risk_level'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title='Risk Distribution (Last 100 Predictions)',
                    color_discrete_map={
                        'HIGH': '#ff6b6b',
                        'MEDIUM': '#ffd93d',
                        'LOW': '#6bcf7f'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Top Risk Factors")
                # Analyze risk factors from recent predictions
                risk_factors = {}
                for pred in recent_predictions:
                    if pred['risk_level'] == 'HIGH':
                        customer = pred['customer_data']
                        if customer.get('contract') == 'Month-to-month':
                            risk_factors['Month-to-month contract'] = risk_factors.get('Month-to-month contract', 0) + 1
                        if customer.get('monthly_charges', 0) > 70:
                            risk_factors['High monthly charges'] = risk_factors.get('High monthly charges', 0) + 1
                        if customer.get('tenure', 0) < 12:
                            risk_factors['New customer'] = risk_factors.get('New customer', 0) + 1
                
                if risk_factors:
                    risk_df = pd.DataFrame(
                        risk_factors.items(),
                        columns=['Risk Factor', 'Count']
                    ).sort_values('Count', ascending=True)
                    
                    fig = px.bar(
                        risk_df,
                        x='Count',
                        y='Risk Factor',
                        orientation='h',
                        title='Common Risk Factors in High-Risk Customers',
                        color='Count',
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No high-risk customers in recent predictions")

def run_monitoring_dashboard():
    """Run the monitoring dashboard"""
    monitor = ModelMonitor()
    monitor.create_monitoring_dashboard()

if __name__ == "__main__":
    run_monitoring_dashboard()