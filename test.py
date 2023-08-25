import streamlit as st
import pandas as pd
import joblib
import toad

# ... (your existing code)

def main():
    st.title("Credit Score Prediction App")
    
    # # Load or train the model
    # model_filename = "credit_score_model.joblib"
    
    # if st.button("Train and Save Model"):
    #     card.fit(train_woe[features_use], train_woe['label'])
    #     joblib.dump(card, model_filename)
    #     st.write("Model trained and saved.")
    
    # if st.button("Load Model"):
    card = joblib.load("model.joblib")
    #     st.write("Model loaded.")
    
    # Take user inputs and predict
    st.header("Credit Score Prediction")
    st.write("Enter the following details to predict your credit score:")
    
    input_data = {
        'LIMIT_BAL': st.number_input("LIMIT_BAL", value=420000.0),
        'EDUCATION': st.number_input("EDUCATION", value=2.0),
        'AGE': st.number_input("AGE", value=37.0),
        'PAY_0': st.number_input("PAY_0", value=0.0),
        'PAY_2': st.number_input("PAY_2", value=0.0),
        'BILL_AMT1': st.number_input("BILL_AMT1", value=10000.0),
        'BILL_AMT2': st.number_input("BILL_AMT2", value=10000.0),
        'PAY_AMT1': st.number_input("PAY_AMT1", value=70000.0),
        'PAY_AMT2': st.number_input("PAY_AMT2", value=1846.0),
    }
    
    input_df = pd.DataFrame(input_data,index=[0])
    #st.write(input_df)
    if st.button("Predict"):
        prediction = card.predict(input_df)
        st.success(f"Predicted Credit Score: {prediction}")

if __name__ == "__main__":
    main()
