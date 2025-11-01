import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
@st.cache_resource
def load_model():
    model = joblib.load('catboost_kidney_model.pkl')
    return model

model = load_model()

# App title
st.title('Kidney Disease Prediction System')
st.markdown("""
This application predicts the likelihood of chronic kidney disease based on clinical parameters.
""")

# Input fields in sidebar
st.sidebar.header('Patient Parameters')

def user_input_features():
    # Numerical parameters
    st.sidebar.subheader('Numerical Values')
    age = st.sidebar.slider('Age', 2, 100, 50)
    bp = st.sidebar.slider('Blood Pressure (mmHg)', 50, 180, 80)
    bgr = st.sidebar.slider('Blood Glucose Random (mg/dL)', 50, 500, 100)
    bu = st.sidebar.slider('Blood Urea (mg/dL)', 10, 300, 30)
    sc = st.sidebar.slider('Serum Creatinine (mg/dL)', 0.4, 20.0, 1.0)
    sod = st.sidebar.slider('Sodium (mEq/L)', 100, 200, 140)
    pot = st.sidebar.slider('Potassium (mEq/L)', 2.0, 8.0, 4.5)
    hg = st.sidebar.slider('Hemoglobin (g/dL)', 3.0, 18.0, 12.0)
    pcv = st.sidebar.slider('Packed Cell Volume (%)', 10, 60, 40)
    wbcc = st.sidebar.slider('WBC Count (cells/cumm)', 2000, 20000, 8000)
    rbcc = st.sidebar.slider('RBC Count (millions/cmm)', 2.0, 8.0, 5.0)
    
    # Categorical parameters (already encoded as 0/1)
    st.sidebar.subheader('Categorical Values')
    rbc = st.sidebar.radio('RBC (0=Normal, 1=Abnormal)', [0, 1])
    pc = st.sidebar.radio('Pus Cell (0=Normal, 1=Abnormal)', [0, 1]) 
    pcc = st.sidebar.radio('Pus Cell Clumps (0=No, 1=Yes)', [0, 1])
    ba = st.sidebar.radio('Bacteria (0=No, 1=Yes)', [0, 1])
    htn = st.sidebar.radio('Hypertension (0=No, 1=Yes)', [0, 1])
    dm = st.sidebar.radio('Diabetes Mellitus (0=No, 1=Yes)', [0, 1])
    cad = st.sidebar.radio('Coronary Artery Disease (0=No, 1=Yes)', [0, 1])
    app = st.sidebar.radio('Appetite (0=Good, 1=Poor)', [0, 1])
    pe = st.sidebar.radio('Pedal Edema (0=No, 1=Yes)', [0, 1])
    ane = st.sidebar.radio('Anemia (0=No, 1=Yes)', [0, 1])
    
    # Specific gravity, albumin, sugar
    sg = st.sidebar.selectbox('Specific Gravity', [1.005, 1.010, 1.015, 1.020, 1.025])
    al = st.sidebar.selectbox('Albumin', [0, 1, 2, 3, 4, 5])
    su = st.sidebar.selectbox('Sugar', [0, 1, 2, 3, 4, 5])
    
    data = {
        'age': age,
        'blood_pressure': bp,
        'specific_gravity': sg,
        'albumin': al,
        'sugar': su,
        'red_blood_cells': rbc,
        'pus_cell': pc,
        'pus_cell_clumps': pcc,
        'bacteria': ba,
        'blood_glucose_random': bgr,
        'blood_urea': bu,
        'serum_creatinine': sc,
        'sodium': sod,
        'potassium': pot,
        'haemoglobin': hg,
        'packed_cell_volume': pcv,
        'white_blood_cell_count': wbcc,
        'red_blood_cell_count': rbcc,
        'hypertension': htn,
        'diabetes_mellitus': dm,
        'coronary_artery_disease': cad,
        'appetite': app,
        'peda_edema': pe,
        'aanemia': ane
    }
    
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Show input parameters
st.subheader('Patient Input Parameters')
st.write(input_df)

# Prediction
if st.button('Get Prediction'):
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)
    
    st.subheader('Result')
    result = 'Kidney Disease Detected' if prediction[0] == 0 else 'No Kidney Disease Detected'
    st.success(f'Diagnosis: {result}')
    
    st.subheader('Probability')
    st.write(f"Probability of Kidney Disease: {proba[0][0]:.2%}")
    st.write(f"Probability of Normal: {proba[0][1]:.2%}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        st.subheader('Feature Importance')
        feat_importance = pd.DataFrame({
            'Feature': input_df.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=feat_importance.head(10))
        st.pyplot(fig)

# Footer
st.markdown("""
---
**Note**: This is only a prediction tool. Please consult a nephrologist for final diagnosis.
""")
