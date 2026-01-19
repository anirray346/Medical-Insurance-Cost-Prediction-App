import streamlit as st
import pandas as pd
import joblib
import numpy as np

# UI Layout
st.set_page_config(page_title="Medical Insurance Prediction", layout="centered")

# Custom CSS for fonts and styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Roboto', sans-serif;
        }
        
        .main-header {
            font-family: 'Roboto', sans-serif;
            font-weight: 700;
            color: #2C3E50;
            text-align: center;
            padding-bottom: 10px;
        }
        
        .sub-header {
            font-family: 'Roboto', sans-serif;
            font-weight: 400;
            color: #34495E;
            text-align: center;
            margin-bottom: 20px;
        }
        
        .stButton>button {
            background-color: #27AE60;
            color: white;
            font-weight: bold;
            border-radius: 5px;
            padding: 10px 20px;
            border: none;
            width: 100%;
        }
        
        .stButton>button:hover {
            background-color: #2ECC71;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, scaler, model_columns
    except FileNotFoundError:
        return None, None, None

model, scaler, model_columns = load_artifacts()

# Display Header Image (Resized and Centered)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("header_img.png", width=300)

st.markdown("<h1 class='main-header'>🏥 Medical Insurance Cost Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Predict insurance charges with advanced Machine Learning models.</p>", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model files not found. Please run 'train_model.py' first to generate the model artifacts.")
    st.stop()

# Navigation Tabs
tab1, tab2 = st.tabs(["💰 Prediction", "📊 Model Comparison"])

with tab1:
    st.markdown("### 📝 Patient Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, help="Age of the primary beneficiary")
        sex = st.selectbox("Sex", ["male", "female"], help="Gender of the policyholder")
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, help="Body Mass Index")
    
    with col2:
        children = st.number_input("Children", min_value=0, max_value=10, value=0, help="Number of children covered")
        smoker = st.selectbox("Smoker", ["yes", "no"], help="Smoking status")
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"], help="Residential area")
    
    st.write("") # Spacer
    submitted = st.button("Predict Costs")

    if submitted:
        # Prepare input data
        input_data = pd.DataFrame({
            'age': [age],
            'bmi': [bmi],
            'children': [children],
            'sex_male': [1 if sex == 'male' else 0],
            'smoker_yes': [1 if smoker == 'yes' else 0],
            'region_northwest': [1 if region == 'northwest' else 0],
            'region_southeast': [1 if region == 'southeast' else 0],
            'region_southwest': [1 if region == 'southwest' else 0]
        })
        
        # Align columns
        for col in model_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[model_columns]
        
        # Scale
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        
        # Display Result
        st.markdown("---")
        st.success(f"Estimated Insurance Charges: **${prediction:,.2f}**")
        
        if smoker == 'yes':
            st.warning("⚠️ Smoking significantly increases insurance costs.")
        if bmi > 30:
            st.info("ℹ️ BMI > 30 is considered obese, which may affect costs.")

with tab2:
    st.markdown("### 📈 Model Performance Comparison")
    
    comparison_data = {
        'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVR', 'KNN', 'Ensemble Learning', 'Best Model (RF Tuned)'],
        'Train RMSE': [6105.55, 494.21, 1934.41, 12568.98, 4488.17, 5487.79, 3309.20],
        'Test RMSE': [5796.28, 6336.72, 4567.96, 12888.96, 5533.37, 6344.19, 4474.28],
        'Train R²': [0.7417, 0.9983, 0.9741, -0.0945, 0.8604, 0.7913, 0.9241],
        'Test R²': [0.7836, 0.7414, 0.8656, -0.0701, 0.8028, 0.7407, 0.8711],
        'Overfitting': ['No', 'Yes', 'Yes', 'No', 'No', 'No', 'No']
    }

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['Test R²'], color='#d1e7dd').format("{:.4f}", subset=['Train R²', 'Test R²']).format("{:.2f}", subset=['Train RMSE', 'Test RMSE']), use_container_width=True)

    st.markdown("""
    #### 💡 Key Insights
    - **Random Forest (Tuned)** achieved the best generalization with a **Test R² of 0.87**.
    - **Linear Regression** is a strong baseline but fails to capture non-linear patterns.
    - **Decision Trees** showed high variance (overfitting) before ensemble methods were applied.
    """)
