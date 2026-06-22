import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder

# Set page config
st.set_page_config(
    page_title="Ayurved",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for white background and white components
st.markdown("""
    <style>
    /* White background for the entire app */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* White background for buttons with black text */
    .stButton > button {
        background-color: #FFFFFF;
        color: #000000;
        border: 1px solid #CCCCCC;
    }
    
    .stButton > button:hover {
        background-color: #F5F5F5;
        color: #000000;
    }
    
    /* White background for selectbox */
    .stSelectbox div[data-testid="stSelectbox"] {
        background-color: #FFFFFF;
    }
    
    .stSelectbox > div {
        background-color: #FFFFFF;
    }
    
    /* White background for info boxes */
    .stInfo {
        background-color: #FFFFFF;
        border: 1px solid #CCCCCC;
        color: #000000;
    }
    
    /* White background for success boxes */
    .stSuccess {
        background-color: #FFFFFF;
        border: 1px solid #CCCCCC;
        color: #000000;
    }
    
    /* Ensure text is readable on white */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #000000;
    }
            
    .stMarkdown{
        color: #000000;        
    }
    
    
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv("AyurGenixAI.csv", encoding="latin1")

data = load_data()

label_encoders = {}
encoded_data = data.copy()

for col in encoded_data.columns:
    le = LabelEncoder()
    encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))
    label_encoders[col] = le

X = encoded_data[["Disease"]]

target_columns = [
    'Symptoms',
    'Diagnosis & Tests',
    'Symptom Severity',
    'Duration of Treatment',
    'Current Medications',
    'Risk Factors',
    'Dietary Habits',
    'Allergies (Food/Env)',
    'Diet and Lifestyle Recommendations',
    'Yoga & Physical Therapy',
    'Medical Intervention',
    'Prevention',
    'Patient Recommendations',
    'Ayurvedic Herbs',
    'Formulation',
    'Ayurvedic  Remedies1',
    'Ayurvedic  Remedies 2',
    'overall'
]

Y = encoded_data[target_columns]

model = MultiOutputClassifier(
    DecisionTreeClassifier(random_state=42)
)

model.fit(X, Y)

disease_list = label_encoders["Disease"].classes_

l,c,r = st.columns([1,4,1])
with c:
    st.title("🏥 Ayurved ")
    st.title(" Ayurvedic solution prediction system ")
    st.title("")

left,center,right = st.columns(3)
with center:
    
    selected_disease = st.selectbox("🔍 Select Disease", disease_list)
    predict_button = st.button("Predict Information", use_container_width=True)

if predict_button:   

    disease_encoded = label_encoders["Disease"].transform([selected_disease]).reshape(-1, 1)

    prediction = model.predict(disease_encoded)

    st.markdown("---")
    st.header(f"📋 Results for: {selected_disease}")

    medical_fields = target_columns[:8]  # Medical fields (excludes the 2 moved to Ayurvedic)
        # Ayurvedic fields with Yoga & Physical Therapy and Diet and Lifestyle Recommendations first, Formulation removed
    ayurvedic_fields = [
            'Yoga & Physical Therapy',
            'Diet and Lifestyle Recommendations',
            'Prevention',
            'Patient Recommendations',
            'Ayurvedic Herbs',
            'Ayurvedic  Remedies1',
            'Ayurvedic  Remedies 2',
            'overall'
        ]

    tab1, tab2 = st.columns(2)
    with tab1:
        st.subheader("🩺 Medical Information")

        for i, col in enumerate(medical_fields):
            idx = target_columns.index(col)

            value = label_encoders[col].inverse_transform(
                    [int(prediction[0][idx])]
                )[0]

            st.markdown(f"### {col}")
            st.info(value)
               
    with tab2:
        st.subheader("🌿 Ayurvedic Information")
        for i, col in enumerate(ayurvedic_fields):
            idx = target_columns.index(col)

            value = label_encoders[col].inverse_transform(
                    [int(prediction[0][idx])]
                )[0]

            st.markdown(f"### {col}")
            st.success(value)
