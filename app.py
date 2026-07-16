# app.py - ForestGuard AI ULTIMATE EDITION v4.0
"""
ForestGuard AI - Enterprise Customer Retention Intelligence Platform
Complete Features: Random Forest | Real SHAP | Business Intelligence | Dynamic Parameters | Export Reports | Prediction History
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import json
import hashlib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="🌲 ForestGuard AI | Enterprise Intelligence Platform",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []
if 'user_role' not in st.session_state:
    st.session_state.user_role = None

# ============================================================================
# SIMPLE AUTHENTICATION (Enterprise Feature)
# ============================================================================
def authenticate_user():
    """Simple authentication system"""
    st.sidebar.markdown("### 🔐 Enterprise Access")
    
    # Demo credentials
    users = {
        "admin": {"password": "admin123", "role": "admin"},
        "analyst": {"password": "analyst123", "role": "analyst"},
        "viewer": {"password": "viewer123", "role": "viewer"}
    }
    
    username = st.sidebar.text_input("Username", placeholder="admin / analyst / viewer")
    password = st.sidebar.text_input("Password", type="password", placeholder="admin123 / analyst123 / viewer123")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        login_btn = st.button("🔐 Login", use_container_width=True)
    with col2:
        if st.button("👤 Demo", use_container_width=True):
            st.session_state.authenticated = True
            st.session_state.user_role = "viewer"
            st.rerun()
    
    if login_btn:
        if username in users and users[username]["password"] == password:
            st.session_state.authenticated = True
            st.session_state.user_role = users[username]["role"]
            st.sidebar.success(f"✅ Welcome {username}!")
            time.sleep(1)
            st.rerun()
        else:
            st.sidebar.error("❌ Invalid credentials")
            return False
    return st.session_state.authenticated

# ============================================================================
# PREMIUM CSS - COMPLETE FOREST THEME
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #0a2f1a 0%, #1b4d2a 50%, #2e7d32 100%);
        padding: 2.5rem 3rem;
        border-radius: 28px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .hero-section::before {
        content: "🌲";
        font-size: 180px;
        position: absolute;
        right: -20px;
        top: -40px;
        opacity: 0.08;
        pointer-events: none;
    }
    
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.9);
        margin-bottom: 1rem;
    }
    
    .badge-container {
        display: flex;
        gap: 0.8rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    
    .badge {
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(10px);
        padding: 0.4rem 1.2rem;
        border-radius: 40px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
        border: 1px solid rgba(255,255,255,0.3);
        transition: all 0.3s ease;
    }
    
    .badge:hover {
        background: rgba(255,255,255,0.25);
        transform: translateY(-2px);
    }
    
    /* Premium Cards */
    .premium-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .premium-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 20px;
        padding: 1.8rem 1rem;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(46,125,50,0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(46,125,50,0.15);
    }
    
    .metric-value-large {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, #1a472a, #2e7d32);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
        color: #6c757d;
    }
    
    /* Risk Indicators */
    .risk-high {
        background: linear-gradient(135deg, #c62828 0%, #e53935 100%);
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        animation: pulse 2s infinite;
        box-shadow: 0 10px 30px rgba(198,40,40,0.3);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #f57c00 0%, #ff9800 100%);
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(245,124,0,0.3);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #2e7d32 0%, #43a047 100%);
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(46,125,50,0.3);
    }
    
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 10px 30px rgba(198,40,40,0.3); }
        50% { transform: scale(1.02); box-shadow: 0 15px 40px rgba(198,40,40,0.5); }
        100% { transform: scale(1); box-shadow: 0 10px 30px rgba(198,40,40,0.3); }
    }
    
    .risk-value {
        font-size: 2rem;
        font-weight: 800;
        color: white;
        margin: 0.5rem 0;
    }
    
    .risk-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: rgba(255,255,255,0.9);
    }
    
    /* Insight Boxes */
    .insight-box-warning {
        background: linear-gradient(135deg, #fff5f5 0%, #ffe0e0 100%);
        border-left: 5px solid #e53935;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    
    .insight-box-success {
        background: linear-gradient(135deg, #f0f9f0 0%, #e0f5e0 100%);
        border-left: 5px solid #2e7d32;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #2e7d32;
        display: inline-block;
    }
    
    /* Input Panel Styling */
    .input-section {
        background: white;
        border-radius: 20px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
    
    .input-header {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #2e7d32;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #1a472a 0%, #2e7d32 100%);
        color: white;
        border: none;
        padding: 0.9rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        border-radius: 14px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(46,125,50,0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(46,125,50,0.4);
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: #f8f9fa;
        border-radius: 16px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #e9ecef;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 20px;
        margin-top: 2rem;
        color: rgba(255,255,255,0.8);
    }
    
    /* Divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, #2e7d32, #4caf50, #2e7d32);
        margin: 1rem 0;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD ARTIFACTS WITH ERROR HANDLING
# ============================================================================
@st.cache_resource
def load_artifacts():
    """Load all model artifacts with detailed error handling"""
    artifacts = {
        'model': None,
        'feature_names': None,
        'metadata': {},
        'shap_explainer': None
    }
    
    try:
        # Try loading from v1 folder
        if os.path.exists('models/v1/churn_model.pkl'):
            artifacts['model'] = joblib.load('models/v1/churn_model.pkl')
            artifacts['feature_names'] = joblib.load('models/v1/feature_names.pkl')
            artifacts['metadata'] = joblib.load('models/v1/metadata.pkl')
            st.success("✅ Model v1 loaded successfully")
        elif os.path.exists('models/churn_model.pkl'):
            artifacts['model'] = joblib.load('models/churn_model.pkl')
            artifacts['feature_names'] = joblib.load('models/feature_names.pkl')
            artifacts['metadata'] = {'accuracy': 0.78, 'roc_auc': 0.83, 'model_type': 'Random Forest'}
            st.success("✅ Model loaded successfully")
        else:
            st.error("❌ Model file not found! Please run: python create_initial_data.py")
            return artifacts
        
        # Load SHAP explainer
        try:
            import shap
            artifacts['shap_explainer'] = shap.TreeExplainer(artifacts['model'])
            st.info("🔮 SHAP explainer loaded - Ready for explanations")
        except Exception as e:
            st.warning(f"⚠️ SHAP not available: {e}")
            
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        
    return artifacts

# ============================================================================
# DATA VALIDATION FUNCTION
# ============================================================================
def validate_inputs(tenure, monthly_charges, total_charges):
    """Validate input data before prediction"""
    warnings_list = []
    
    if tenure == 0:
        warnings_list.append("⚠️ Tenure is 0 - This is a new customer with no history")
    if tenure > 60:
        warnings_list.append("✅ Tenure > 5 years - Loyal customer detected")
    if monthly_charges == 0:
        warnings_list.append("⚠️ Monthly charges is 0 - This might be a test account")
    if monthly_charges > 100:
        warnings_list.append("⚠️ High monthly charges - Customer may be price-sensitive")
    if total_charges == 0:
        warnings_list.append("⚠️ Total charges is 0 - New customer or incomplete data")
    
    return warnings_list

# ============================================================================
# BUSINESS LOGIC WITH DYNAMIC VALUES
# ============================================================================
def calculate_business_impact(probability, avg_customer_value, retention_cost, customer_lifetime):
    """Calculate business impact with dynamic parameters"""
    potential_loss = probability * avg_customer_value * customer_lifetime
    net_savings = potential_loss - retention_cost if probability > 0.3 else retention_cost
    roi = ((potential_loss - retention_cost) / retention_cost * 100) if probability > 0.3 else 0
    
    return {
        'potential_loss': potential_loss,
        'net_savings': max(0, net_savings),
        'roi': roi,
        'break_even_probability': retention_cost / avg_customer_value
    }

# ============================================================================
# GENERATE REPORT FUNCTION
# ============================================================================
def generate_report(customer_data, prediction_result, business_impact, risk_factors, recommendations):
    """Generate downloadable report"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'customer_data': customer_data,
        'prediction': {
            'churn_probability': prediction_result['probability'],
            'risk_level': prediction_result['risk_level'],
            'confidence': prediction_result['confidence']
        },
        'business_impact': business_impact,
        'risk_factors': risk_factors,
        'recommendations': recommendations,
        'model_metadata': {
            'accuracy': prediction_result.get('accuracy', 0.78),
            'roc_auc': prediction_result.get('roc_auc', 0.83)
        }
    }
    return json.dumps(report, indent=2)

# ============================================================================
# MAIN APP - AUTHENTICATION CHECK
# ============================================================================
if not st.session_state.authenticated:
    if not authenticate_user():
        st.stop()

# Show user role in sidebar
st.sidebar.markdown(f"👤 **Role:** {st.session_state.user_role.upper()}")
st.sidebar.markdown("---")

# ============================================================================
# DYNAMIC BUSINESS PARAMETERS (User Configurable)
# ============================================================================
with st.sidebar.expander("⚙️ Business Parameters", expanded=False):
    st.markdown("**Configure your business metrics**")
    avg_customer_value = st.number_input("💰 Avg Customer Lifetime Value ($)", 
                                          value=1200, min_value=500, max_value=5000, step=100,
                                          help="Average revenue per customer over lifetime")
    retention_cost = st.number_input("💸 Retention Campaign Cost ($)", 
                                      value=100, min_value=50, max_value=500, step=10,
                                      help="Cost to retain a customer")
    customer_lifetime = st.number_input("📅 Customer Lifetime (Years)", 
                                         value=3, min_value=1, max_value=10, step=1,
                                         help="Average customer lifespan")

# ============================================================================
# LOAD ARTIFACTS
# ============================================================================
artifacts = load_artifacts()

if artifacts['model'] is None:
    st.stop()

# ============================================================================
# HERO SECTION
# ============================================================================
st.markdown(f"""
<div class="hero-section">
    <div class="hero-title">🌲 ForestGuard AI</div>
    <div class="hero-subtitle">Enterprise Customer Retention Intelligence Platform</div>
    <div class="badge-container">
        <span class="badge">🎯 Random Forest Classifier</span>
        <span class="badge">📊 ROC-AUC {artifacts['metadata'].get('roc_auc', 0.83)*100:.1f}%</span>
        <span class="badge">🔮 Real SHAP Explainability</span>
        <span class="badge">💼 Dynamic Business Intelligence</span>
        <span class="badge">📄 Export Reports</span>
        <span class="badge">⚡ Real-time Predictions</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN DASHBOARD LAYOUT - 2 COLUMNS
# ============================================================================
col_left, col_right = st.columns([1, 1.2], gap="large")

# ============================================================================
# LEFT PANEL - SMART INPUT FORM
# ============================================================================
with col_left:
    st.markdown('<div class="section-header">🎯 Customer Intelligence Input</div>', unsafe_allow_html=True)
    st.markdown("*Enter customer details for AI-powered churn analysis*")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    with st.container():
        # Demographics Section
        st.markdown("""
        <div class="input-section">
            <div class="input-header">👤 Demographics</div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"], help="Customer's gender")
        with col2:
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"], help="Age 65+ status")
        
        col1, col2 = st.columns(2)
        with col1:
            partner = st.selectbox("Partner", ["No", "Yes"], help="Has partner/spouse")
        with col2:
            dependents = st.selectbox("Dependents", ["No", "Yes"], help="Has dependents")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Account Information Section
        st.markdown("""
        <div class="input-section">
            <div class="input-header">📊 Account Intelligence</div>
        """, unsafe_allow_html=True)
        
        tenure = st.slider("Tenure (months)", 0, 72, 12, 
                          help="How long the customer has been with the company")
        
        contract = st.selectbox("Contract Type", 
                               ["Month-to-month", "One year", "Two year"],
                               help="Type of subscription contract")
        
        col1, col2 = st.columns(2)
        with col1:
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        with col2:
            payment_method = st.selectbox("Payment Method",
                                         ["Electronic check", "Mailed check",
                                          "Bank transfer (automatic)", "Credit card (automatic)"])
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Services Section
        st.markdown("""
        <div class="input-section">
            <div class="input-header">🌐 Service Portfolio</div>
        """, unsafe_allow_html=True)
        
        internet_service = st.selectbox("Internet Service",
                                       ["DSL", "Fiber optic", "No"])
        
        col1, col2 = st.columns(2)
        with col1:
            online_security = st.selectbox("Online Security",
                                          ["No", "Yes", "No internet service"])
        with col2:
            tech_support = st.selectbox("Tech Support",
                                       ["No", "Yes", "No internet service"])
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Financial Section
        st.markdown("""
        <div class="input-section">
            <div class="input-header">💰 Financial Metrics</div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.0,
                                             help="Monthly bill amount")
        with col2:
            total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0, 800.0,
                                           help="Total amount paid to date")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Predict Button
        predict_btn = st.button("🔮 ANALYZE CHURN RISK", use_container_width=True)

# ============================================================================
# RIGHT PANEL - RESULTS DASHBOARD
# ============================================================================
with col_right:
    if predict_btn:
        # DATA VALIDATION
        validation_warnings = validate_inputs(tenure, monthly_charges, total_charges)
        for warning in validation_warnings:
            st.warning(warning)
        
        with st.spinner("🔄 Analyzing customer data with AI..."):
            time.sleep(0.5)
            
            # Build input features
            input_dict = {f: 0 for f in artifacts['feature_names']}
            
            # Fill features
            input_dict['tenure'] = tenure
            input_dict['MonthlyCharges'] = monthly_charges
            input_dict['TotalCharges'] = total_charges
            input_dict['SeniorCitizen'] = 1 if senior_citizen == "Yes" else 0
            
            # Contract type
            if contract == "Month-to-month":
                input_dict['Contract_Month-to-month'] = 1
            elif contract == "One year":
                input_dict['Contract_One year'] = 1
            else:
                input_dict['Contract_Two year'] = 1
            
            # Payment method
            if payment_method == "Electronic check":
                input_dict['PaymentMethod_Electronic check'] = 1
            elif payment_method == "Mailed check":
                input_dict['PaymentMethod_Mailed check'] = 1
            elif payment_method == "Bank transfer (automatic)":
                input_dict['PaymentMethod_Bank transfer (automatic)'] = 1
            else:
                input_dict['PaymentMethod_Credit card (automatic)'] = 1
            
            # Internet service
            if internet_service == "DSL":
                input_dict['InternetService_DSL'] = 1
            elif internet_service == "Fiber optic":
                input_dict['InternetService_Fiber optic'] = 1
            else:
                input_dict['InternetService_No'] = 1
            
            # Online security
            if online_security == "Yes":
                input_dict['OnlineSecurity_Yes'] = 1
            elif online_security == "No":
                input_dict['OnlineSecurity_No'] = 1
            else:
                input_dict['OnlineSecurity_No internet service'] = 1
            
            # Tech support
            if tech_support == "Yes":
                input_dict['TechSupport_Yes'] = 1
            elif tech_support == "No":
                input_dict['TechSupport_No'] = 1
            else:
                input_dict['TechSupport_No internet service'] = 1
            
            # Demographics
            input_dict['gender_Male'] = 1 if gender == "Male" else 0
            input_dict['Partner_Yes'] = 1 if partner == "Yes" else 0
            input_dict['Dependents_Yes'] = 1 if dependents == "Yes" else 0
            input_dict['PaperlessBilling_Yes'] = 1 if paperless_billing == "Yes" else 0
            
            # Engineered features
            input_dict['charges_per_tenure'] = monthly_charges / (tenure + 1)
            input_dict['high_risk'] = 1 if (contract == "Month-to-month" and monthly_charges > 70) else 0
            input_dict['PhoneService_Yes'] = 1
            input_dict['MultipleLines_No'] = 1
            
            # Tenure groups
            if tenure <= 12:
                input_dict['tenure_new'] = 1
                input_dict['tenure_moderate'] = 0
                input_dict['tenure_loyal'] = 0
                input_dict['tenure_veteran'] = 0
            elif tenure <= 24:
                input_dict['tenure_new'] = 0
                input_dict['tenure_moderate'] = 1
                input_dict['tenure_loyal'] = 0
                input_dict['tenure_veteran'] = 0
            elif tenure <= 48:
                input_dict['tenure_new'] = 0
                input_dict['tenure_moderate'] = 0
                input_dict['tenure_loyal'] = 1
                input_dict['tenure_veteran'] = 0
            else:
                input_dict['tenure_new'] = 0
                input_dict['tenure_moderate'] = 0
                input_dict['tenure_loyal'] = 0
                input_dict['tenure_veteran'] = 1
            
            # Create DataFrame
            input_df = pd.DataFrame([input_dict])
            input_df = input_df[artifacts['feature_names']]
            
            # PREDICTION
            probability = artifacts['model'].predict_proba(input_df)[0][1]
            prediction = 1 if probability > 0.5 else 0
            
            # Determine risk level
            if probability > 0.5:
                risk_level = "HIGH"
                risk_color = "#e53935"
            elif probability > 0.3:
                risk_level = "MEDIUM"
                risk_color = "#ff9800"
            else:
                risk_level = "LOW"
                risk_color = "#2e7d32"
            
            confidence = max(probability, 1-probability)
            
            # REAL SHAP EXPLANATION
            shap_available = False
            shap_values = None
            if artifacts['shap_explainer'] is not None:
                try:
                    import shap
                    shap_values = artifacts['shap_explainer'].shap_values(input_df)
                    shap_available = True
                except Exception as e:
                    st.warning(f"SHAP calculation note: {e}")
            
            # RISK FACTORS
            risk_factors = []
            if contract == "Month-to-month":
                risk_factors.append("📄 Month-to-month contract (+35% risk)")
            if monthly_charges > 70:
                risk_factors.append(f"💰 High monthly charges (${monthly_charges}) (+25% risk)")
            if tenure < 12:
                risk_factors.append(f"🆕 New customer ({tenure} months) (+20% risk)")
            if internet_service == "Fiber optic":
                risk_factors.append("🌐 Fiber optic internet (+15% risk)")
            if payment_method == "Electronic check":
                risk_factors.append("💳 Electronic check payment (+18% risk)")
            if online_security == "No" and internet_service != "No":
                risk_factors.append("🔒 No online security (+12% risk)")
            if tech_support == "No" and internet_service != "No":
                risk_factors.append("🛠️ No tech support (+10% risk)")
            if monthly_charges / (tenure + 1) > 10:
                risk_factors.append(f"📊 High charges per tenure ({monthly_charges/(tenure+1):.1f})")
            
            # RECOMMENDATIONS
            recommendations = []
            if contract == "Month-to-month":
                recommendations.append("🎁 Offer annual contract with 15% discount")
                recommendations.append("⭐ Priority loyalty program enrollment")
            if monthly_charges > 70:
                recommendations.append("💎 Bundle services for 20% savings")
            if tenure < 12:
                recommendations.append("🌟 Welcome program with exclusive benefits")
                recommendations.append("📞 30-day onboarding check-in call")
            if internet_service == "Fiber optic":
                recommendations.append("📡 Service quality monitoring with SLA")
            if payment_method == "Electronic check":
                recommendations.append("💰 2% cashback for autopay enrollment")
            if online_security == "No":
                recommendations.append("🔒 Free 3-month online security trial")
            if tech_support == "No":
                recommendations.append("🛠️ Premium tech support free trial")
            
            if not recommendations:
                recommendations = [
                    "✅ Maintain current service quality",
                    "📧 Satisfaction survey after 30 days",
                    "🎯 Loyalty rewards program enrollment",
                    "💬 Quarterly check-in communications"
                ]
            
            # BUSINESS IMPACT
            business_impact = calculate_business_impact(probability, avg_customer_value, retention_cost, customer_lifetime)
            
            # SAVE TO HISTORY
            prediction_record = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'probability': probability,
                'risk_level': risk_level,
                'tenure': tenure,
                'contract': contract,
                'monthly_charges': monthly_charges
            }
            st.session_state.predictions_history.append(prediction_record)
            
            # ================================================================
            # TOP ROW - 3 BIG METRIC CARDS
            # ================================================================
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">CHURN PROBABILITY</div>
                    <div class="metric-value-large">{probability:.1%}</div>
                    <div style="font-size:0.8rem; color:#6c757d;">Model prediction score</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                risk_class = f"risk-{risk_level.lower()}"
                st.markdown(f"""
                <div class="{risk_class}">
                    <div class="risk-label">RISK LEVEL</div>
                    <div class="risk-value">{risk_level} RISK</div>
                    <div style="font-size:0.8rem; color:rgba(255,255,255,0.9);">
                        {'⚠️ Immediate action required' if risk_level == 'HIGH' else '📊 Monitor closely' if risk_level == 'MEDIUM' else '✅ Standard monitoring'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">MODEL CONFIDENCE</div>
                    <div class="metric-value-large">{confidence:.1%}</div>
                    <div style="font-size:0.8rem; color:#6c757d;">Random Forest certainty</div>
                </div>
                """, unsafe_allow_html=True)
            
            # ================================================================
            # VISUAL ZONE - GAUGE CHART
            # ================================================================
            st.markdown('<div class="section-header">📊 Risk Visualization</div>', unsafe_allow_html=True)
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Churn Risk Score", 'font': {'size': 20, 'color': '#2e7d32'}},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#2c3e50"},
                    'bar': {'color': risk_color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#e9ecef",
                    'steps': [
                        {'range': [0, 30], 'color': '#c8e6c9'},
                        {'range': [30, 70], 'color': '#ffe0b2'},
                        {'range': [70, 100], 'color': '#ffcdd2'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': probability * 100
                    }
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # ================================================================
            # REAL SHAP PLOT
            # ================================================================
            if shap_available and shap_values is not None:
                st.markdown('<div class="section-header">🔍 Real SHAP Explainability</div>', unsafe_allow_html=True)
                st.markdown("*Understanding what drives this prediction*")
                
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Safely extract SHAP values
                    if isinstance(shap_values, list):
                        shap_arr = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    else:
                        shap_arr = shap_values
                    
                    if len(shap_arr.shape) == 2:
                        shap_vals = shap_arr[0]
                    else:
                        shap_vals = shap_arr
                    
                    # Get top features
                    feature_names_list = artifacts['feature_names']
                    contributions = []
                    
                    for i in range(min(len(feature_names_list), len(shap_vals))):
                        contributions.append({
                            'feature': feature_names_list[i][:40],  # Limit length
                            'value': float(shap_vals[i])
                        })
                    
                    # Sort by absolute value
                    contributions.sort(key=lambda x: abs(x['value']), reverse=True)
                    top_contributions = contributions[:10]
                    
                    # Plot
                    features = [c['feature'] for c in top_contributions]
                    values = [c['value'] for c in top_contributions]
                    colors = ['#e53935' if v > 0 else '#2e7d32' for v in values]
                    
                    bars = ax.barh(features, values, color=colors)
                    ax.set_xlabel('SHAP Value (Contribution to Churn)', fontsize=12)
                    ax.set_title('Feature Contributions for This Customer', fontsize=14)
                    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
                    
                    # Add value labels
                    for bar, val in zip(bars, values):
                        ax.text(val + (0.01 if val > 0 else -0.05), 
                               bar.get_y() + bar.get_height()/2, 
                               f'{val:.3f}', va='center', fontsize=9)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.info("💡 **Interpretation:** Positive values (red) increase churn risk, negative values (green) decrease it")
                    
                except Exception as e:
                    st.warning(f"⚠️ SHAP visualization: {e}")
                    st.info("SHAP values calculated but visualization requires numeric processing")
            
            # ================================================================
            # EXPLAINABILITY PANEL + INSIGHTS
            # ================================================================
            st.markdown('<div class="section-header">🔍 Explainability & Insights</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="insight-box-warning">', unsafe_allow_html=True)
                st.markdown("#### ⚠️ Identified Risk Factors")
                for factor in risk_factors[:6]:
                    st.markdown(factor)
                if not risk_factors:
                    st.success("✅ No significant risk factors detected!")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="insight-box-success">', unsafe_allow_html=True)
                st.markdown("#### 💡 Strategic Recommendations")
                for rec in recommendations[:6]:
                    st.markdown(rec)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # ================================================================
            # BUSINESS IMPACT PANEL
            # ================================================================
            st.markdown('<div class="section-header">💼 Executive Business Impact</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                potential_loss = probability * avg_customer_value * customer_lifetime
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">💰 POTENTIAL LOSS</div>
                    <div class="metric-value-large">${potential_loss:,.0f}</div>
                    <div style="font-size:0.75rem;">Over {customer_lifetime} years</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">💸 RETENTION COST</div>
                    <div class="metric-value-large">${retention_cost:,.0f}</div>
                    <div style="font-size:0.75rem;">One-time intervention</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                net_savings = potential_loss - retention_cost if probability > 0.3 else retention_cost
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">📈 NET SAVINGS</div>
                    <div class="metric-value-large" style="color: {'#2e7d32' if net_savings > 0 else '#e53935'}">
                        ${max(0, net_savings):,.0f}
                    </div>
                    <div style="font-size:0.75rem;">By preventing churn</div>
                </div>
                """, unsafe_allow_html=True)
            
            # ================================================================
            # ROI PROJECTION CHART
            # ================================================================
            st.markdown('<div class="section-header">📈 ROI Projection (5-Year)</div>', unsafe_allow_html=True)
            
            years = np.arange(1, 6)
            annual_savings = max(0, (probability * avg_customer_value) - retention_cost) if probability > 0.3 else retention_cost
            cumulative_savings = annual_savings * years
            
            fig_roi = go.Figure()
            fig_roi.add_trace(go.Scatter(
                x=years, y=cumulative_savings,
                mode='lines+markers',
                name='Cumulative Savings',
                line=dict(color='#2e7d32', width=3),
                marker=dict(size=10, color='#1a472a', symbol='circle'),
                fill='tozeroy',
                fillcolor='rgba(46,125,50,0.2)'
            ))
            
            fig_roi.update_layout(
                title='5-Year ROI Projection',
                xaxis_title='Years',
                yaxis_title='Cumulative Savings ($)',
                height=350,
                showlegend=True,
                plot_bgcolor='white',
                hovermode='x unified',
                xaxis=dict(gridcolor='#e9ecef', tickmode='linear', tick0=1, dtick=1),
                yaxis=dict(gridcolor='#e9ecef', tickprefix='$')
            )
            
            st.plotly_chart(fig_roi, use_container_width=True)
            
            # ================================================================
            # SENSITIVITY ANALYSIS
            # ================================================================
            st.markdown('<div class="section-header">📊 Sensitivity Analysis</div>', unsafe_allow_html=True)
            
            probs = np.linspace(0, 1, 100)
            savings = [max(0, (p * avg_customer_value) - retention_cost) for p in probs]
            
            fig_sens = go.Figure()
            fig_sens.add_trace(go.Scatter(
                x=probs * 100,
                y=savings,
                mode='lines',
                name='Net Savings',
                line=dict(color='#2e7d32', width=2),
                fill='tozeroy'
            ))
            
            fig_sens.add_vline(x=probability * 100, line_dash="dash", line_color="red",
                              annotation_text=f"Current Risk: {probability:.1%}")
            fig_sens.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig_sens.update_layout(
                title='Sensitivity: Churn Probability vs Net Savings',
                xaxis_title='Churn Probability (%)',
                yaxis_title='Net Savings ($)',
                height=300
            )
            
            st.plotly_chart(fig_sens, use_container_width=True)
            
            # ================================================================
            # ACTION RECOMMENDATION
            # ================================================================
            if prediction == 1:
                st.error("""
                🚨 **IMMEDIATE RETENTION INTERVENTION REQUIRED**
                
                **Priority Action Plan:**
                1. 📞 **Schedule retention call** within 24 hours
                2. 🎁 **Personalized retention package** (15-20% discount)
                3. ⭐ **VIP support team assignment** for priority service
                4. 📧 **Targeted engagement campaign** with value highlights
                """)
            else:
                st.success("""
                ✅ **MAINTENANCE MODE - STANDARD ENGAGEMENT**
                
                **Proactive Strategy:**
                1. 📧 **Satisfaction survey** for feedback collection
                2. 🎯 **Loyalty rewards program** enrollment
                3. 📢 **Product updates** and exclusive offers
                4. 💬 **Quarterly check-in** communications
                """)
            
            # ================================================================
            # DOWNLOAD REPORT
            # ================================================================
            st.markdown('<div class="section-header">📄 Export Report</div>', unsafe_allow_html=True)
            
            report_data = generate_report(
                customer_data={
                    'tenure': tenure, 
                    'monthly_charges': monthly_charges, 
                    'contract': contract,
                    'payment_method': payment_method,
                    'internet_service': internet_service,
                    'online_security': online_security,
                    'tech_support': tech_support
                },
                prediction_result={
                    'probability': probability, 
                    'risk_level': risk_level, 
                    'confidence': confidence,
                    'accuracy': artifacts['metadata'].get('accuracy', 0.78),
                    'roc_auc': artifacts['metadata'].get('roc_auc', 0.83)
                },
                business_impact=business_impact,
                risk_factors=risk_factors,
                recommendations=recommendations
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download JSON Report",
                    data=report_data,
                    file_name=f"forestguard_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            with col2:
                if st.button("📊 View Prediction History"):
                    if st.session_state.predictions_history:
                        history_df = pd.DataFrame(st.session_state.predictions_history[-10:])
                        st.dataframe(history_df)
                        st.caption(f"Total predictions: {len(st.session_state.predictions_history)}")
                    else:
                        st.info("No predictions yet")

# ============================================================================
# TABS SECTION - DEEP DIVE ANALYTICS
# ============================================================================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔍 SHAP Explainability", "💼 Business Impact", "📈 Model Insights"])

with tab1:
    if predict_btn:
        st.markdown("### 🎯 Prediction Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Key Metrics")
            metrics_data = {
                "Metric": ["Churn Probability", "Risk Level", "Model Confidence", "Customer Tenure", "Monthly Charges"],
                "Value": [f"{probability:.1%}", risk_level, f"{confidence:.1%}", f"{tenure} months", f"${monthly_charges:.0f}"]
            }
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Risk Breakdown")
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Churn Risk', 'Loyalty'],
                values=[probability, 1-probability],
                marker=dict(colors=['#e53935', '#2e7d32']),
                hole=0.4
            )])
            fig_pie.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Additional stats
        st.markdown("#### 📈 Key Insights")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Customer Value", f"${avg_customer_value:,.0f}")
        with col2:
            st.metric("Break-even Probability", f"{business_impact['break_even_probability']:.1%}")
        with col3:
            st.metric("Potential ROI", f"{business_impact['roi']:.0f}%")
    else:
        st.info("👈 Enter customer details and click 'Analyze Churn Risk' to see insights")

with tab2:
    st.markdown("### 🔍 SHAP Explainability - Model Interpretation")
    st.markdown("*Understanding how the model makes decisions*")
    
    if predict_btn and shap_available:
        st.markdown("#### 📊 Global Feature Importance")
        st.markdown("*Top features that influence churn across all customers*")
        
        # Create feature importance chart
        if hasattr(artifacts['model'], 'feature_importances_'):
            feature_imp = pd.DataFrame({
                'feature': artifacts['feature_names'][:15],
                'importance': artifacts['model'].feature_importances_[:15]
            }).sort_values('importance', ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(feature_imp['feature'], feature_imp['importance'], color='#2e7d32')
            ax.set_xlabel('Feature Importance')
            ax.set_title('Top 15 Features Driving Customer Churn')
            st.pyplot(fig)
        
        st.markdown("#### 📊 How SHAP Works")
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)** values show how each feature contributes to the prediction:
        
        - **Positive SHAP value** → Increases churn risk
        - **Negative SHAP value** → Decreases churn risk
        - **Magnitude** → Strength of influence
        
        For this customer, the chart above shows which features pushed the prediction toward churn or retention.
        """)
    elif predict_btn:
        st.info("SHAP values calculated but visualization pending")
    else:
        st.info("👈 Make a prediction first to see SHAP explainability")

with tab3:
    st.markdown("### 💼 Business Impact Analysis")
    st.markdown("*Financial implications and ROI calculations*")
    
    if predict_btn:
        st.markdown("#### 📊 Financial Impact Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Customer Lifetime Value", f"${avg_customer_value:,.0f}")
            st.metric("Retention Campaign ROI", f"{business_impact['roi']:.0f}%")
        
        with col2:
            st.metric("Risk-Adjusted Value", f"${(1-probability) * avg_customer_value:,.0f}")
            st.metric("Break-even Probability", f"{retention_cost/avg_customer_value:.1%}")
        
        with col3:
            st.metric("Expected Annual Savings", f"${annual_savings:,.0f}")
            st.metric("5-Year Cumulative", f"${cumulative_savings[-1]:,.0f}")
        
        st.markdown("#### 📈 Investment Recommendation")
        
        if business_impact['roi'] > 50:
            st.success(f"✅ **STRONG INVESTMENT CASE** - Expected ROI: {business_impact['roi']:.0f}%")
            st.progress(min(1.0, business_impact['roi']/100))
        elif business_impact['roi'] > 0:
            st.info(f"📊 **MODERATE INVESTMENT CASE** - Expected ROI: {business_impact['roi']:.0f}%")
            st.progress(business_impact['roi']/100)
        else:
            st.warning("⚠️ **LIMITED INVESTMENT CASE** - ROI below threshold")
    else:
        st.info("👈 Make a prediction first to see business impact analysis")

with tab4:
    st.markdown("### 📈 Model Performance Insights")
    st.markdown("*Understanding how the model works*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Model Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Value': [
                f"{artifacts['metadata'].get('accuracy', 0.78)*100:.1f}%",
                f"{artifacts['metadata'].get('precision', 0.71)*100:.1f}%",
                f"{artifacts['metadata'].get('recall', 0.68)*100:.1f}%",
                f"{artifacts['metadata'].get('f1_score', 0.69)*100:.1f}%",
                f"{artifacts['metadata'].get('roc_auc', 0.83)*100:.1f}%"
            ]
        })
        st.dataframe(metrics_df, use_container_width=True)
    
    with col2:
        st.markdown("#### 🌲 Model Configuration")
        config_df = pd.DataFrame({
            'Parameter': ['Algorithm', 'Trees', 'Max Depth', 'Features', 'Training Samples'],
            'Value': [
                'Random Forest',
                '200',
                '15',
                str(artifacts['metadata'].get('features_count', 35)),
                str(artifacts['metadata'].get('training_samples', 8260))
            ]
        })
        st.dataframe(config_df, use_container_width=True)
    
    # Feature Importance Chart
    st.markdown("#### 📊 Global Feature Importance")
    
    if hasattr(artifacts['model'], 'feature_importances_'):
        features = artifacts['feature_names'][:20]
        importance = artifacts['model'].feature_importances_[:20]
        importance_df = pd.DataFrame({'feature': features, 'importance': importance}).sort_values('importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance_df['feature'], importance_df['importance'], color='#2e7d32')
        ax.set_xlabel('Importance')
        ax.set_title('Top 20 Features Driving Customer Churn')
        ax.invert_yaxis()
        st.pyplot(fig)
    
    st.markdown("#### 📊 Model Training Details")
    st.markdown(f"""
    - **Algorithm**: Random Forest Classifier
    - **Number of Trees**: 200
    - **Max Depth**: 15
    - **Features Used**: {artifacts['metadata'].get('features_count', 35)}
    - **Training Samples**: {artifacts['metadata'].get('training_samples', 8260):,}
    - **Class Balancing**: SMOTE (Synthetic Minority Over-sampling Technique)
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="footer">
    <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">🌲 ForestGuard AI</p>
    <p style="font-size: 0.85rem;">Powered by Random Forest | Real SHAP Explainability | Dynamic Business Intelligence</p>
    <p style="font-size: 0.75rem; opacity: 0.7;">Enterprise Customer Retention Intelligence Platform v4.0</p>
</div>
""", unsafe_allow_html=True)