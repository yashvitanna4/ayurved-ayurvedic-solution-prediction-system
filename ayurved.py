

import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Ayurved",
    page_icon="🏥",
    layout="wide"
)


st.markdown("""
<style>
.stApp {
    background-color: white;
}

h1, h2, h3, h4, h5, h6, p, label, span {
    color: black !important;
}

.stButton > button {
    background-color: white;
    color: black;
    border: 1px solid #cccccc;
}

.stButton > button:hover {
    background-color: #f5f5f5;
}

div[data-testid="stSelectbox"] {
    background-color: white;
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
    encoded_data[col] = le.fit_transform(
        encoded_data[col].astype(str)
    )
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


disease_list = sorted(
    data["Disease"].dropna().unique()
)

symptom_list = sorted(
    data["Symptoms"].dropna().unique()
)


l, c, r = st.columns([1, 2, 1])

with c:

    st.title("🏥 Ayurved")
    st.subheader("Ayurvedic Solution Prediction System")

    st.image(
        "https://www.aimilpharma.life/cdn/shop/articles/blog-banner.jpg?v=1602927357",
        width=700
    )
    st.markdown("""
<div style="color:black;">

### 🌿 What is Ayurveda?
Ayurveda is an ancient Indian system of medicine that promotes
balance between body, mind and spirit.

It focuses on:
- 🌿 Natural herbs
- 🥗 Healthy diet
- 🧘 Yoga and meditation
- 😴 Proper sleep
- 🚶 Healthy lifestyle

According to Ayurveda, good health depends upon maintaining
balance among the three doshas:

- Vata
- Pitta
- Kapha

Ayurvedic treatments aim to restore balance naturally.

</div>
""", unsafe_allow_html=True)
st.subheader("🔍 How would you like to continue?")

search_type = st.radio(
    "Choose an option",
    ["Disease", "Symptoms"],
    horizontal=True
)

prediction = None
selected_disease = None

if search_type == "Disease":

    selected_disease = st.selectbox(
        "Select Disease",
        disease_list
    )

    predict_button = st.button(
        "Predict Information",
        use_container_width=True
    )

    if predict_button:

        disease_encoded = label_encoders["Disease"].transform(
            [selected_disease]
        ).reshape(-1, 1)

        prediction = model.predict(
            disease_encoded
        )


else:

    selected_symptom = st.selectbox(
        "Select Symptom",
        symptom_list
    )

    predict_button = st.button(
        "Find Disease & Predict",
        use_container_width=True
    )

    if predict_button:

        matching_rows = data[
            data["Symptoms"] == selected_symptom
        ]

        if len(matching_rows) > 0:

            selected_disease = matching_rows.iloc[0]["Disease"]

            st.success(
                f"Possible Disease: {selected_disease}"
            )

            disease_encoded = label_encoders["Disease"].transform(
                [selected_disease]
            ).reshape(-1, 1)

            prediction = model.predict(
                disease_encoded
            )

        else:
            st.error("No disease found.")


if prediction is not None:

    st.markdown("---")

    st.header(
        f"📋 Results for: {selected_disease}"
    )

    medical_fields = [
        'Symptoms',
        'Diagnosis & Tests',
        'Symptom Severity',
        'Duration of Treatment',
        'Current Medications',
        'Risk Factors',
        'Dietary Habits',
        'Allergies (Food/Env)',
        'Medical Intervention'
    ]

    ayurvedic_fields = [
        'Yoga & Physical Therapy',
        'Diet and Lifestyle Recommendations',
        'Prevention',
        'Patient Recommendations',
        'Ayurvedic Herbs',
        'Formulation',
        'Ayurvedic  Remedies1',
        'Ayurvedic  Remedies 2',
        'overall'
    ]

    col1, col2 = st.columns(2)

 
    with col1:

        st.subheader("🩺 Medical Information")

        for field in medical_fields:

            idx = target_columns.index(field)

            value = label_encoders[field].inverse_transform(
                [int(prediction[0][idx])]
            )[0]

            st.markdown(f"### {field}")
            st.info(value)


    with col2:

        st.subheader("🌿 Ayurvedic Information")

        for field in ayurvedic_fields:

            idx = target_columns.index(field)

            value = label_encoders[field].inverse_transform(
                [int(prediction[0][idx])]
            )[0]

            st.markdown(f"### {field}")
            st.success(value)

    st.balloons()

