import streamlit as st
import pandas as pd
import joblib

# Chargement du modèle et de l'encodeur
model = joblib.load('model_svm.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# Titre de l'application
st.title('Ophthalmic Genetic Diagnosis Prediction')

# Création des champs de saisie pour les entrées
age = st.text_input('Age', '0')
unilateral_cataract = st.text_input('Unilateral Cataract', '0')
bilateral_cataract = st.text_input('Bilateral Cataract', '0')
corneal_opacity = st.text_input('Corneal Opacity', '0')
retinal_affliction = st.text_input('Retinal Affliction', '0')

# Bouton de prédiction
if st.button('Predict PAX6 Status'):
    # Préparation des données pour la prédiction
    input_data = pd.DataFrame([[age, unilateral_cataract, bilateral_cataract, corneal_opacity, retinal_affliction]],
                              columns=['Age', 'Unilateral Cataract', 'Bilateral Cataract', 'Corneal Opacity', 'Retinal Affliction'])
    input_data_encoded = label_encoder.transform(input_data)

    # Prédiction
    prediction = model.predict(input_data_encoded)
    st.write(f'Predicted PAX6 Status: {prediction[0]}')
