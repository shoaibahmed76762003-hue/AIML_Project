import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==========================================
# 1. DATA COLLECTION & PREPROCESSING
# ==========================================

@st.cache_data
def load_or_generate_data():
    """
    Loads data from CSV if it exists, otherwise generates synthetic data 
    matching the Project PPT methodology.
    """
    dataset_path = "financial_data.csv"
    
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
    else:
        # Generating synthetic data for demonstration
        np.random.seed(42)
        n = 1000
        
        income = np.random.randint(30000, 200000, n)
        # Expenses and EMI are logically tied to income
        expenses = income * np.random.uniform(0.3, 0.8, n) 
        emi_amount = income * np.random.uniform(0.0, 0.4, n)
        
        credit_utilization = np.random.uniform(0.0, 1.0, n)
        inflation_rate = np.random.uniform(3.0, 8.0, n)
        interest_rate = np.random.uniform(5.0, 15.0, n)

        df = pd.DataFrame({
            'income': income,
            'expenses': expenses,
            'emi_amount': emi_amount,
            'credit_utilization': credit_utilization,
            'inflation_rate': inflation_rate,
            'interest_rate': interest_rate
        })

    # ==========================================
    # 2. FEATURE ENGINEERING
    # ==========================================
    # Create the features mentioned in the presentation
    df['savings_ratio'] = (df['income'] - df['expenses'] - df['emi_amount']) / df['income']
    df['expense_ratio'] = df['expenses'] / df['income']
    df['emi_burden'] = df['emi_amount'] / df['income']

    # Handle any potential negatives in savings ratio
    df['savings_ratio'] = df['savings_ratio'].apply(lambda x: x if x > 0 else 0)

    # ==========================================
    # 3. TARGET VARIABLE CREATION
    # ==========================================
    # Define Stress Levels: 0 = Low, 1 = Medium, 2 = High
    # (If using a real dataset, this target column would already exist)
    if 'stress_level' not in df.columns:
        conditions = [
            (df['savings_ratio'] < 0.1) | (df['emi_burden'] > 0.45) | (df['credit_utilization'] > 0.8),
            (df['savings_ratio'] < 0.25) | (df['emi_burden'] > 0.3) | (df['credit_utilization'] > 0.5)
        ]
        choices = [2, 1] # 2: High, 1: Medium
        df['stress_level'] = np.select(conditions, choices, default=0) # 0: Low

    return df

# ==========================================
# 4. MODEL BUILDING & EVALUATION
# ==========================================

@st.cache_resource
def train_models(df):
    # Features and Target
    features = ['income', 'expenses', 'emi_amount', 'credit_utilization', 
                'inflation_rate', 'interest_rate', 'savings_ratio', 
                'expense_ratio', 'emi_burden']
    
    X = df[features]
    y = df['stress_level']

    # Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize Models
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }

    metrics = {}
    trained_models = {}

    # Train and Evaluate
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        metrics[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, average='weighted')
        }
        trained_models[name] = model

    return trained_models, scaler, metrics, features

# Load data and train models
df = load_or_generate_data()
models, scaler, metrics, feature_cols = train_models(df)

# ==========================================
# 5. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Financial Stress Index App", layout="wide")

st.title(" AI-Based Financial Stress Index")
st.markdown("**Domain:** Banking & Finance | **Goal:** SDG 8 – Decent Work and Economic Growth")
st.markdown("---")

# Sidebar for Model Analytics (Satisfies Methodology Display)
with st.sidebar:
    st.header(" Model Evaluation")
    st.write("Performance metrics based on testing data:")
    
    selected_model_name = st.selectbox("Choose a Model for Prediction", list(models.keys()))
    
    st.divider()
    st.subheader(f"{selected_model_name} Metrics")
    st.write(f"**Accuracy:** {metrics[selected_model_name]['Accuracy']:.2%}")
    st.write(f"**Precision:** {metrics[selected_model_name]['Precision']:.2%}")
    st.write(f"**Recall:** {metrics[selected_model_name]['Recall']:.2%}")
    st.write(f"**F1-Score:** {metrics[selected_model_name]['F1-Score']:.2%}")
    
    st.divider()
    st.write("Developed by **SHOAIB AHMED**")
    st.write("SRN: PES1PG25CA205")

# Main Interface: User Inputs
st.subheader("👤 Enter Individual Financial Data")

col1, col2, col3 = st.columns(3)

with col1:
    user_income = st.number_input("Monthly Income (₹)", min_value=5000, max_value=1000000, value=50000, step=1000)
    user_expenses = st.number_input("Monthly Expenses (₹) (Excluding EMI)", min_value=0, max_value=1000000, value=30000, step=1000)

with col2:
    user_emi = st.number_input("Monthly Loan/EMI Repayments (₹)", min_value=0, max_value=500000, value=5000, step=500)
    user_credit_util = st.slider("Credit Card Utilization (%)", 0, 100, 30) / 100.0

with col3:
    user_inflation = st.number_input("Current Inflation Rate (%)", min_value=0.0, max_value=20.0, value=6.0, step=0.1)
    user_interest = st.number_input("Average Loan Interest Rate (%)", min_value=0.0, max_value=30.0, value=9.0, step=0.1)

# ==========================================
# 6. PREDICTION & SUGGESTIONS
# ==========================================

if st.button("Predict Financial Stress Level", type="primary"):
    
    # Calculate Engineered Features for the user
    user_savings_ratio = (user_income - user_expenses - user_emi) / user_income
    user_savings_ratio = max(0, user_savings_ratio) # prevent negative ratio dropping below 0 logically
    user_expense_ratio = user_expenses / user_income
    user_emi_burden = user_emi / user_income
    
    # Format input for the model
    input_data = pd.DataFrame([[
        user_income, user_expenses, user_emi, user_credit_util, 
        user_inflation, user_interest, user_savings_ratio, 
        user_expense_ratio, user_emi_burden
    ]], columns=feature_cols)
    
    input_scaled = scaler.transform(input_data)
    
    # Predict using selected model
    model = models[selected_model_name]
    prediction = model.predict(input_scaled)[0]
    
    # Display Results
    st.markdown("---")
    st.subheader("Financial Health Report")
    
    if prediction == 0:
        st.success("### Status: LOW FINANCIAL STRESS 🟢")
        st.write("**Analysis & Suggestions:**")
        st.write("- Your debt-to-income and savings ratios look healthy.")
        st.write("- **Next Step:** Consider investing surplus savings into appreciating assets like mutual funds or fixed deposits.")
        st.write("- Ensure your emergency fund has 6 months' worth of expenses.")
        
    elif prediction == 1:
        st.warning("### Status: MEDIUM FINANCIAL STRESS 🟡")
        st.write("**Analysis & Suggestions:**")
        st.write("- You are experiencing some financial pressure, likely from moderate debt or lower savings.")
        st.write("- **Next Step:** Apply the 50/30/20 budget rule (50% needs, 30% wants, 20% savings).")
        st.write("- Try to minimize discretionary spending and avoid taking on any new loans.")
        
    else:
        st.error("### Status: HIGH FINANCIAL STRESS 🔴")
        st.write("**Analysis & Suggestions:**")
        st.write("- Your financial data indicates severe instability (high EMI burden, low savings, or high credit utilization).")
        st.write("- **Action Required:** Halt all non-essential credit card usage.")
        st.write("- Consider debt consolidation to lower your monthly interest rates.")
        st.write("- Reevaluate your core expenses immediately to free up cash flow.")

    # Show the generated ratios to the user contextually
    st.divider()
    st.write(f"**Your Extracted Financial Markers:**")
    st.write(f"• Savings Ratio: {user_savings_ratio:.2%}")
    st.write(f"• EMI Burden: {user_emi_burden:.2%}")
