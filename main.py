import streamlit as st
import joblib
import pandas
import sklearn

# Load your model and label encoder
svm_model = joblib.load('svm_model.pkl')
le = joblib.load('label_encoder.pkl')

def main():
    st.title("ML Model Prediction")

    # Creating text input boxes for each feature
    age = st.number_input("Age", min_value=0.5, max_value=100.0, step=0.1)
    cataracte_unilaterale = st.selectbox("Cataracte Unilatérale", [0, 1])  # Assuming binary input: 0 for No, 1 for Yes
    cataracte_bilaterale = st.selectbox("Cataracte Bilatérale", [0, 1])
    opacite_cornee = st.selectbox("Opacité Cornée", [0, 1])
    atteinte_retine = st.selectbox("Atteinte Rétine", [0, 1])

    # Button to make prediction
    if st.button('Predict'):
        # Prepare the feature vector for prediction
        features = [age, cataracte_unilaterale, cataracte_bilaterale, opacite_cornee, atteinte_retine]
        
        # Make a prediction
        prediction = svm_model.predict([features])
        prediction_label = le.inverse_transform(prediction)

        # Determine the text to display based on the prediction
        if prediction_label[0] == 1:
            result_text = "Le patient a le gène PAX6"
        else:
            result_text = "Le patient n'a pas le gène PAX6"

        # Display the result
        st.write(f"Prediction: {result_text}")

if __name__ == '__main__':
    main()
