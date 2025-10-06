import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration MUST be the first Streamlit command
st.set_page_config(page_title="Brain Tumor Prediction", layout="wide", initial_sidebar_state="expanded")

# --- PENAMBAHAN BARU 1: Inisialisasi Riwayat (History) ---
# Cek apakah 'history' sudah ada di memori sesi, jika belum, buat list kosong
if 'history' not in st.session_state:
    st.session_state.history = []

# =============================================================================
# BACKEND FUNCTIONS
# =============================================================================
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('brain_tumor_stacking_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_model_and_scaler()

default_values = {
    'Mean': 6.535339, 'Variance': 619.587845, 'Standard Deviation': 24.891522, 'Entropy': 0.109059,
    'Skewness': 4.276477, 'Kurtosis': 18.900575, 'Contrast': 98.613971, 'Energy': 0.293314,
    'ASM': 0.086033, 'Homogeneity': 0.530941, 'Dissimilarity': 4.473346, 'Correlation': 0.981939, 'Coarseness': 0.0
}
# =============================================================================
# SIDEBAR (FOR INPUTS)
# =============================================================================
st.sidebar.title("🧠 Data Input Panel")
st.sidebar.markdown("Enter the values for each medical attribute below.")

feature_names = ['Mean', 'Variance', 'Standard Deviation', 'Entropy', 'Skewness', 'Kurtosis', 'Contrast', 'Energy', 'ASM', 'Homogeneity', 'Dissimilarity', 'Correlation', 'Coarseness']
help_texts = [
    "The average pixel intensity level.", "The variation or spread of intensity levels.",
    "The square root of Variance, also measuring the spread.", "The degree of randomness or complexity in the image.",
    "The degree of asymmetry in the intensity distribution.", "The 'peakedness' of the intensity distribution.",
    "A measure of the local variations between pixels.", "A measure of the uniformity or regularity of a pattern.",
    "Another name for Energy, measuring uniformity.", "A measure of the texture's smoothness.",
    "A measure of texture roughness (opposite of Homogeneity).", "A measure of the presence of linear patterns.",
    "A measure of the scale or 'graininess' of the texture."
]

user_inputs = {}
for feature, help_text in zip(feature_names, help_texts):
    default_val = default_values.get(feature, 0.0)
    user_inputs[feature] = st.sidebar.number_input(f"Value for '{feature}'", value=default_val, format="%.5f", key=feature, help=help_text)

# Prediction button is placed in the sidebar
predict_button = st.sidebar.button('**Get Prediction**', type="primary", use_container_width=True)

# =============================================================================
# MAIN PAGE DISPLAY
# =============================================================================
st.title("Brain Tumor Classification Prediction")
st.warning("**Disclaimer:** Please consult a doctor for an accurate diagnosis. This tool is intended to serve as a second opinion for doctors to improve accuracy and efficiency in detecting brain tumors.", icon="⚠️")

# Creating TABS
tab_prediction, tab_attributes, tab_project = st.tabs(["📈 **Prediction Result**", "📚 **Attribute Descriptions**", "ℹ️ **About the Project**"])

with tab_prediction:
    st.header("Prediction Result from the AI Model")
    if predict_button:
        empty_columns = [
            feature_name for feature_name, value in user_inputs.items() 
            if value == 0.0 and feature_name != 'Coarseness'
        ]
        
        if len(empty_columns) > 0:
            st.error(f"**Warning:** Please fill in all attributes (except Coarseness) for an accurate prediction. The following attributes are still set to 0.0:", icon="❗")
            for column in empty_columns:
                st.markdown(f"- **{column}**")
        else:
            if model is not None and scaler is not None:
                input_df = pd.DataFrame([user_inputs])
                input_df = input_df[feature_names]
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)
                prediction_proba = model.predict_proba(input_scaled)

                if prediction[0] == 1:
                    result_text = "Brain Tumor Indicated"
                    confidence = prediction_proba[0][1] * 100
                    st.error(f"### Result: {result_text}", icon="⚠️")
                else:
                    result_text = "No Brain Tumor Indicated"
                    confidence = prediction_proba[0][0] * 100
                    st.success(f"### Result: {result_text}", icon="✅")

                st.metric(label="Model Confidence Level", value=f"{confidence:.2f}%")
                st.progress(int(confidence))

                # --- PENAMBAHAN BARU 2: Simpan Hasil ke Riwayat ---
                history_entry = {
                    "inputs": input_df.copy(),
                    "result": result_text,
                    "confidence": confidence,
                    "id": len(st.session_state.history) + 1
                }
                st.session_state.history.insert(0, history_entry)

                with st.expander("View Processed Data Details"):
                    st.write("Initial Input Data:")
                    st.dataframe(input_df)
                    st.write("Data After Scaling (as seen by the model):")
                    st.dataframe(pd.DataFrame(input_scaled, columns=feature_names))
            else:
                st.error("Model could not be loaded. Please check your .pkl files.")
    else:
        st.info("Please enter all attribute values in the left panel and click the 'Get Prediction' button.")

    # --- PENAMBAHAN BARU 3: Tampilkan Riwayat Prediksi ---
    st.divider()
    st.header("Prediction History (This Session)")

    # Tombol untuk membersihkan riwayat
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun() # Muat ulang halaman agar tampilan riwayat kosong

    if not st.session_state.history:
        st.info("No prediction history for this session yet.")
    else:
        # Tampilkan setiap entri riwayat
        for entry in st.session_state.history:
            with st.expander(f"History #{entry['id']} - Result: **{entry['result']}** ({entry['confidence']:.2f}%)"):
                st.write("Input Data Used:")
                st.dataframe(entry['inputs'])

with tab_attributes:
    st.header("Attribute Glossary")
    st.write("Here are simple descriptions for each attribute used by the model:")
    for feature, help_text in zip(feature_names, help_texts):
        st.info(f"**{feature}**: {help_text}")

with tab_project:
    st.header("About")
    st.markdown("""
    This website is an implementation of the research project **[COMPARISON OF LOGISTIC REGRESSION, SUPPORT VECTOR MACHINE, AND K-NEAREST NEIGHBOR USING ENSEMBLE METHOD IN BRAIN TUMOR CLASSIFICATION]**.
    Its purpose is to assist in brain tumor classification based on 13 features extracted from medical imagery.

    ### Model Used
    The "brain" of this website is an **Ensemble Stacking** model, an advanced technique that combines the strengths of three base models to enhance prediction accuracy and reliability:
    1.  **K-Nearest Neighbors (KNN)**: With n_neighbors=11 and 'manhattan' metric.
    2.  **Logistic Regression (ALR)**: With C=0.000082.
    3.  **Support Vector Machine (SVM)**: With a 'poly' kernel and degree=3.

    ### Created by:
    **Azmi Muhammad Padhil**
    
    **Ally Muchlas**
                
    **Ferdi Setiawan**
                
    **Muhammad Iqbal Arsyad. H** 
    
    **Pasma Azzahra**
                
    **Tyara Hestyani Putri** """)
