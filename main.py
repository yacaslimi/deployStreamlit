import streamlit as st
import joblib
import numpy as np

# Load your model and label encoder
svm_model = joblib.load('svm_model.pkl')
le = joblib.load('label_encoder.pkl')

def main():
    st.title("ML Model Prediction")

    with st.form("prediction_form"):
        st.markdown("### Please enter the patient details:")

        # Using columns for a better layout
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", min_value=0.5, max_value=100.0, step=0.1, value=30.0)
            cataracte_unilaterale = st.selectbox("Cataracte Unilatérale", [0, 1])

        with col2:
            cataracte_bilaterale = st.selectbox("Cataracte Bilatérale", [0, 1])
            opacite_cornee = st.selectbox("Opacité Cornée", [0, 1])
            atteinte_retine = st.selectbox("Atteinte Rétine", [0, 1])

        submitted = st.form_submit_button("Predict")
        if submitted:
            # Prepare the feature vector for prediction
            features = np.array([age, cataracte_unilaterale, cataracte_bilaterale, opacite_cornee, atteinte_retine]).reshape(1, -1)
            
            # Make a prediction
            prediction = svm_model.predict(features)
            prediction_label = le.inverse_transform(prediction)
            
            # Display the result
            st.markdown(f"#### Prediction: {prediction_label[0]}")

if __name__ == '__main__':
    main()
