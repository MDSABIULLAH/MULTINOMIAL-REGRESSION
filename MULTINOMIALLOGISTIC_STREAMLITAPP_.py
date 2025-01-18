import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
import joblib
import pickle

# Function to load the trained model
def load_model():
    with open('multinomial.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Data preprocessing function
def preprocess_data(dataframe):
    expected_columns = [
        'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate', 'installment',
        'annual_inc', 'dti', 'inq_last_6mths', 'open_acc', 'revol_bal', 'total_acc',
        'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
        'last_pymnt_amnt', 'delinq_2yrs', 'pub_rec', 'out_prncp', 'out_prncp_inv',
        'total_rec_late_fee', 'home_ownership', 'verification_status', 'purpose'
    ]
    
    dataframe = dataframe[expected_columns]

    if 'term' in dataframe.columns:
        dataframe['term'] = dataframe['term'].str.replace(' months', '').astype(int)
    if 'int_rate' in dataframe.columns:
        dataframe['int_rate'] = dataframe['int_rate'].str.replace('%', '').astype(float)

    normal_variance_pipeline = joblib.load('normal_variance.joblib')
    low_variance_pipeline = joblib.load('low_variance.joblib')
    category_pipeline = joblib.load('category.joblib')

    low_variance_cols = ['delinq_2yrs', 'pub_rec', 'out_prncp', 'out_prncp_inv', 'total_rec_late_fee']
    normal_variance_cols = dataframe.select_dtypes(exclude='object').columns.difference(low_variance_cols)

    transformed_numerical_normal = normal_variance_pipeline.transform(dataframe[normal_variance_cols])
    transformed_numerical_low = low_variance_pipeline.transform(dataframe[low_variance_cols])
    transformed_categorical = category_pipeline.transform(dataframe[['home_ownership', 'verification_status', 'purpose']])

    transformed_data = np.hstack([
        transformed_numerical_normal, 
        transformed_numerical_low, 
        transformed_categorical
    ])

    return transformed_data

# Function to save results to MySQL database using SQLAlchemy's create_engine
def save_to_database(df, host, user, password, database):
    try:
        # Create the database engine using SQLAlchemy
        engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')
        
        # Store the DataFrame to the database
        df.to_sql('Loan_Predictions_new_1', con=engine, if_exists='replace', index=False)
        
        st.success("Data has been successfully stored in the database.")
    
    except Exception as e:
        st.error(f"An error occurred while connecting to the database: {e}")




# Main function for Streamlit app
def main():
    st.title('Loan Status Prediction App')
    
    st.markdown("""
    ## Introduction
    This app uses a machine learning model to predict the loan status based on user-provided information.
    Users can upload a CSV file, and after prediction, the data can be stored in a MySQL database.

    ## Instructions
    1. Enter your MySQL database credentials below.
    2. Prepare a CSV file containing loan data.
    3. Upload the file and view predictions.
    4. Store results in the database.
    """)

    # User inputs for database credentials
    st.sidebar.header("Database Credentials")
    host = st.sidebar.text_input("Host", value="localhost")
    user = st.sidebar.text_input("User")
    password = st.sidebar.text_input("Password", type="password")
    database = st.sidebar.text_input("Database")

    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(input_df.head())

        try:
            transformed_data = preprocess_data(input_df)
            model = load_model()
            predictions = model.predict(transformed_data)
            result_df = input_df.copy()
            result_df['Predicted_Loan_Status'] = predictions

            st.write("Input Data with Predictions:")
            st.dataframe(result_df)

            if st.button("Save Predictions to Database"):
                save_to_database(result_df, host, user, password, database)

        except Exception as e:
            st.error(f"An error occurred while processing the data: {e}")
            st.write("Please ensure your input data matches the expected format.")

if __name__ == '__main__':
    main()
