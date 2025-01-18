# 1.	Business Problem
# 1.1.	What is the business objective?
# 1.2.	Are there any constraints?

# 2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image:
# Make a table as shown above and provide information about the features such as its data type and its relevance to the model building. And if not relevant, provide reasons and a description of the feature.
# Using Python codes perform:
# 3.	Data Pre-processing
# 3.1 Data Cleaning, Feature Engineering, etc.
# 3.2 Outlier Treatment.
# 4.	Exploratory Data Analysis (EDA):
# 4.1.	Summary.
# 4.2.	Univariate analysis.
# 4.3.	Bivariate analysis.

# 5.	Model Building
# 5.1	Build the model on the scaled data (try multiple options).
# 5.2	Build a Logistic Regression model.
# 5.3	Train and test the model and compare accuracies by building a confusion matrix, and plotting ROC and AUC curves.
# 5.4	Briefly explain the model output in the documentation. 

# 6.	Write about the benefits/impact of the solution - in what way does the business (client) benefit from the solution provided?






# Problem Statement: 

# 1.	You work for a consumer finance company that specializes in lending loans to urban customers. When the company receives a loan application, the company has to make a decision for loan approval based on the applicant’s profile. Two types of risks are associated with the bank’s decision: 
# •	If the applicant is likely to repay the loan, then not approving the loan results in a loss of business to the company 
# •	If the applicant is not likely to repay the loan, i.e. he/she is likely to default, then approving the loan may lead to a financial loss for the company 

# The data given below contains information about past loan applicants and whether they ‘defaulted’4 or not. The aim is to identify patterns that indicate if a person is likely to default, which may be used for taking actions such as denying the loan, reducing the amount of the loan, lending (for risky applicants) at a higher interest rate, etc. 

# In this case study, you will use EDA to understand how consumer attributes and loan attributes influence the tendency of default. 

# When a person applies for a loan, there are two types of decisions that could be taken by the company: 

# 1. Loan accepted: If the company approves the loan, there are 3 possible scenarios described below: 
# •	Fully paid: Applicant has fully paid the loan (the principal and the interest rate) 
# •	Current: Applicant is in the process of paying the installments, i.e., the tenure of the loan is not yet completed. These candidates are not labeled as 'defaulted'. 
# •	Charged-off: Applicant has not paid the installments in due time for a long period of time, i.e. he/she has defaulted on the loan  
# 2. Loan rejected: The company had rejected the loan (because the candidate does not meet their requirements etc.). Since the loan was rejected, there is no transactional history of those applicants with the company and so this data is not available with the company (and thus in this dataset)

# Like most other lending companies, lending loans to ‘risky’ applicants is the largest source of financial loss (called credit loss). Credit loss is the amount of money lost by the lender when the borrower refuses to pay or runs away with the money owed. In other words, borrowers who default cause the largest amount of loss to the lenders. In this case, the customers labeled as 'charged-off' are the 'defaulters'.  
# If one can identify these risky loan applicants, then such loans can be reduced thereby cutting down the amount of credit loss. 
# In other words, the company wants to understand the driving factors (or driver variables) behind loan default, i.e. the variables which are strong indicators of default.  The company can utilize this knowledge for its portfolio and risk assessment.  

# Perform Multinomial regression on the dataset in which loan_status is the output (Y) variable and it has three levels in it. 









"""

Business Objective : Identify factors that predict loan default risk and minimize credit loss by 25% through informed lending decisions.
    
Business Constraint : Maintain a balanced loan portfolio with a maximum of 15% high-risk loans while complying with regulatory requirements and ensuring fair lending practices.


Success Criteria :

    Business Success Criteria : Reduce default rate by 10% and increase overall loan profitability by 20%
        
    Ml Success Criteria : Develop a multinomial regression model with 85% accuracy in predicting loan status.
        
    Economic Success Criteria : Decrease credit loss by 30% and improve risk-adjusted return on loan portfolio by 15%.



"""



"""

Data Dictionary :
    
id: A unique LC assigned ID for the loan listing
member_id: A unique LC assigned ID for the borrower member
loan_amnt: The listed amount of the loan applied for by the borrower
funded_amnt: The total amount committed to that loan
funded_amnt_inv: The total amount committed by investors for that loan
term: The number of payments on the loan. Values are in months and can be either 36 or 60
int_rate: Interest rate on the loan
installment: The monthly payment owed by the borrower if the loan originates
grade: LC assigned loan grade
sub_grade: LC assigned loan subgrade
emp_title: The job title supplied by the borrower when applying for the loan
emp_length: Employment length in years
home_ownership: The home ownership status provided by the borrower during registration or obtained from the credit report
annual_inc: The self-reported annual income provided by the borrower during registration
verification_status: Indicates if income was verified by LC, not verified, or if the income source was verified
issue_d: The month which the loan was funded
loan_status: Current status of the loan
pymnt_plan: Indicates if a payment plan has been put in place for the loan
url: URL for the LC page with listing data
desc: Loan description provided by the borrower
purpose: A category provided by the borrower for the loan request
title: The loan title provided by the borrower
zip_code: The first 3 numbers of the zip code provided by the borrower in the loan application
addr_state: The state provided by the borrower in the loan application
dti: A ratio calculated using the borrower's total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower's self-reported monthly income
delinq_2yrs: The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years
earliest_cr_line: The month the borrower's earliest reported credit line was opened
inq_last_6mths: The number of inquiries in past 6 months (excluding auto and mortgage inquiries)
mths_since_last_delinq: The number of months since the borrower's last delinquency
mths_since_last_record: The number of months since the last public record
open_acc: The number of open credit lines in the borrower's credit file
pub_rec: Number of derogatory public records
revol_bal: Total credit revolving balance
revol_util: Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit
total_acc: The total number of credit lines currently in the borrower's credit file
initial_list_status: The initial listing status of the loan
out_prncp: Remaining outstanding principal for total amount funded
out_prncp_inv: Remaining outstanding principal for portion of total amount funded by investors
total_pymnt: Payments received to date for total amount funded
total_pymnt_inv: Payments received to date for portion of total amount funded by investors
total_rec_prncp: Principal received to date
total_rec_int: Interest received to date
total_rec_late_fee: Late fees received to date
recoveries: Post charge off gross recovery
collection_recovery_fee: Post charge off collection fee
last_pymnt_d: Last month payment was received
last_pymnt_amnt: Last total payment amount received
next_pymnt_d: Next scheduled payment date
last_credit_pull_d: The most recent month LC pulled credit for this loan
collections_12_mths_ex_med: Number of collections in 12 months excluding medical collections
mths_since_last_major_derog: Months since most recent 90-day or worse rating
policy_code: Publicly available policy_code=1, new products not publicly available policy_code=2

"""



# Importing necessary libraries for data manipulation, visualization, and modeling.
import pandas as pd  # For data manipulation and analysis.
import numpy as np  # For numerical computations.
import matplotlib.pyplot as plt  # For plotting graphs.
import seaborn as sns  # For enhanced visualizations with matplotlib.
from sqlalchemy import create_engine, text  # For SQL database connections.
from urllib.parse import quote  # For escaping special characters in URLs.
from sklearn.pipeline import Pipeline  # For constructing pipelines for data processing.
from feature_engine.outliers import Winsorizer  # For handling outliers in data.
from sklearn.impute import SimpleImputer  # For handling missing data.
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder  # For scaling and encoding.
from sklearn.compose import ColumnTransformer  # For applying transformations on multiple column sets.
import joblib  # For model persistence (saving and loading).
import pickle  # For serializing Python objects.
from sklearn.linear_model import LogisticRegression  # For logistic regression models.
from sklearn.metrics import accuracy_score  # For evaluating model accuracy with specific metrics.
from sklearn.model_selection import train_test_split, GridSearchCV  # For splitting data and hyperparameter optimization.




# Reading the CSV file into a DataFrame.
loan = pd.read_csv("C:/Users/user/Desktop/data science assignment question/MULTINOMIAL LOGISTIC REGRESSION/loan.csv")
loan.head()  # Display the first few rows of the DataFrame.
loan.info()  # Output concise summary information about the DataFrame.





# Database connection parameters.
user = 'root'
pw = '12345678'
db = 'univ_db'

# Create a database engine connection using SQLAlchemy.
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# Save the DataFrame to a SQL table named 'loan_data_table', replacing it if it already exists.
loan.to_sql('loan_data_table', if_exists='replace', index=False, con=engine)

# SQL query to retrieve all records from the 'loan_data_table'.
sql = 'select * from loan_data_table'

# Read the SQL query results into a DataFrame.
loan_df = pd.read_sql_query(text(sql), engine.connect())

# Check the first and last few rows of the DataFrame.
loan_df.head()
loan_df.tail()






# Count different values in the 'loan_status' column.
loan_df['loan_status'].value_counts()

# Check for missing values in each column.
loan_df.isna().sum()

# Drop columns with any missing values.
loan_df_nclean = loan_df.dropna(axis=1)
loan_df_nclean.head()
loan_df_nclean.info()

# Check to confirm there are no remaining missing values.
loan_df_nclean.isna().sum()

# Drop irrelevant or non-essential columns from the DataFrame.
loan_df_clean = loan_df_nclean.drop([
    'id', 'member_id', 'grade', 'sub_grade', 'issue_d', 'pymnt_plan', 'url', 'zip_code',
    'addr_state', 'earliest_cr_line', 'initial_list_status', 'recoveries',
    'collection_recovery_fee', 'application_type'
], axis=1)

# Remove 'months' from 'term' column and convert to integer type.
loan_df_clean['term'] = loan_df_clean['term'].str.replace(' months', '').astype(int)
# Remove '%' from 'int_rate' column and convert to float type.
loan_df_clean['int_rate'] = loan_df_clean['int_rate'].str.replace('%', '').astype(float)

loan_df_clean.head()
loan_df_clean.info()

# Display the count of each 'loan_status'.
loan_df_clean['loan_status'].value_counts()

# Compute correlation matrix for numerical columns to identify any columns that could be dropped.
loan_df_clean.corr(numeric_only=True)

# Drop columns with undefined (NaN) correlation values.
loan_df_clean_final = loan_df_clean.drop(['policy_code', 'acc_now_delinq', 'delinq_amnt'], axis=1)
loan_df_clean_final.head()
loan_df_clean_final.info()

# Recompute correlation matrix to confirm the changes.
loan_df_clean_final.corr(numeric_only=True)

# Separating input (X) and output (Y) data.
loan_y = loan_df_clean_final[['loan_status']]
loan_y.head()
loan_y.info()

loan_x = loan_df_clean_final.drop(['loan_status'], axis=1)
loan_x.head()
loan_x.info()

# Further separate categorical and numerical columns in the input set.
loan_x_category = loan_x.select_dtypes(include='object')
loan_x_category.head()
loan_x_category.info()

loan_x_numerical = loan_x.select_dtypes(exclude='object')
loan_x_numerical.head()
loan_x_numerical.info()

# Identify numerical columns with low variance.
low_variance = loan_x_numerical[['delinq_2yrs', 'pub_rec', 'out_prncp', 'out_prncp_inv', 'total_rec_late_fee']]
low_variance.head()
low_variance.info()

# Columns with normal variance after removing low variance ones.
normal_variance = loan_x_numerical.drop(['delinq_2yrs', 'pub_rec', 'out_prncp', 'out_prncp_inv', 'total_rec_late_fee'], axis=1)
normal_variance.head()
normal_variance.info()

# Visual check for outliers in numerical columns using box plots.
loan_x_numerical.plot(kind='box', subplots=True, sharey=False, figsize=(18, 10))
plt.subplots_adjust(wspace=0.75)  # Adjust whitespace between plots for better readability.
plt.xticks(rotation=90)  # Rotate x-ticks for better readability.
plt.show()

# Create a pipeline for processing numerical data with normal variance.
pipeline_num_norm = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),  # Impute missing values with mean.
    ('winsor', Winsorizer(capping_method='iqr', tail='both', fold=1.5)),  # Handle outliers using winsorization.
    ('scale', MinMaxScaler())  # Scale data to a 0-1 range.
])

# Apply the numerical pipeline transformations.
transformed1 = ColumnTransformer(transformers=[
    ('trans1', pipeline_num_norm, normal_variance.columns)
])

fitted_data1 = transformed1.fit(normal_variance)

# Save the transformed pipeline for future use.
joblib.dump(fitted_data1, 'normal_variance.joblib')

# Create a new DataFrame with transformed numerical data.
dataframe1 = pd.DataFrame(fitted_data1.transform(normal_variance), columns=normal_variance.columns)
dataframe1.head()
dataframe1.info()

# Create a pipeline for processing numerical data with low variance.
pipeline_num_low = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),  # Impute missing values with mean.
    ('winsor', Winsorizer(capping_method='gaussian', tail='both', fold=0.05)),  # Handle outliers with tighter winsorization.
    ('scale', MinMaxScaler())  # Scale data to a 0-1 range.
])

# Apply the numerical pipeline transformations for low variance data.
transformed2 = ColumnTransformer(transformers=[
    ('trans2', pipeline_num_low, low_variance.columns)
])

fitted_data2 = transformed2.fit(low_variance)

# Save the transformed pipeline for future use.
joblib.dump(fitted_data2, 'low_variance.joblib')

# Create a new DataFrame with transformed low variance numerical data.
dataframe2 = pd.DataFrame(fitted_data2.transform(low_variance), columns=low_variance.columns)
dataframe2.head()
dataframe2.info()

# Create a pipeline for processing categorical data.
pipeline_cat = Pipeline(steps=[
    ('encoding', OneHotEncoder(sparse_output=False))  # Apply one-hot encoding.
])

# Apply transformations to the categorical data.
transformed3 = ColumnTransformer(transformers=[
    ('trans3', pipeline_cat, loan_x_category.columns)
])

fitted_data3 = transformed3.fit(loan_x_category)

# Save the transformed pipeline for future use.
joblib.dump(fitted_data3, 'category.joblib')

# Create a new DataFrame with transformed categorical data.
dataframe3 = pd.DataFrame(fitted_data3.transform(loan_x_category), columns=fitted_data3.get_feature_names_out())
dataframe3.head()
dataframe3.info()

# Combine all transformed data into one DataFrame.
loan_final_clean_df = pd.concat([dataframe1, dataframe2, dataframe3], axis=1)
loan_final_clean_df.head()
loan_final_clean_df.info()










# Model building phase using Logistic Regression.
basic_model = LogisticRegression(multi_class='multinomial', solver='newton-cg')

# Fit the model with the cleaned and transformed data.
basic_model.fit(loan_final_clean_df, loan_y)

# Predicting loan status using the fitted model.
pred_basic_model = basic_model.predict(loan_final_clean_df)
print(pred_basic_model)

# Calculating and print the accuracy of the model.
acc1 = accuracy_score(loan_y, pred_basic_model)
print(acc1)

# Split data into training and test sets for model evaluation.
X_train, X_test, Y_train, Y_test = train_test_split(loan_final_clean_df, loan_y, random_state=0, stratify=loan_y)

# Train the logistic regression model on the training data.
train_model = LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(X_train, Y_train)

# Predict using the trained model on the training data.
pred_train = train_model.predict(X_train)
print(pred_train)

# Calculate and print accuracy score on the training data.
train_accuracy = accuracy_score(Y_train, pred_train)
print(train_accuracy)

# Predict using the trained model on the test data.
pred_test = train_model.predict(X_test)

# Calculate and print accuracy score on the test data.
test_accuracy = accuracy_score(Y_test, pred_test)
print(test_accuracy)









# Instantiate a logistic regression model for hyperparameter tuning.
logistic_model = LogisticRegression(multi_class='multinomial')

# Define the hyperparameter grid to search.
param_grid = [
    {
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],  # Penalty type.
        'C': np.logspace(-4, 4, 20),  # Regularization strength.
        'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],  # Algorithm for optimization.
        'max_iter': [100, 1000, 2500, 5000]  # Maximum number of iterations.
    }
]

# Configure the grid search for finding the best hyperparameters.
gridmodel = GridSearchCV(logistic_model, param_grid=param_grid, cv=3, verbose=True, n_jobs=-1)

# Fit the grid search model to find the best parameters.
best_clf = gridmodel.fit(X_train, Y_train)
 
# Output the best estimator found by the grid search.
best_clf.best_estimator_

# Print the accuracy of the best model on the training set.
print(f'Accuracy on training set: {best_clf.score(X_train, Y_train):.3f}')

print(f'Accuracy on test set: {best_clf.score(X_test, Y_test):.3f}')

# Fit the grid search on the full dataset to finalize the best model.
best_clf1 = gridmodel.fit(loan_final_clean_df, loan_y)
  
# Retrieve the best model estimator from the grid search.
best_model = best_clf.best_estimator_

# Print the accuracy of the best model on full data and test set.
print(f'Accuracy on full data: {best_clf1.score(loan_final_clean_df, loan_y):.3f}')
print(f'Accuracy on test set: {best_clf1.score(X_test, Y_test):.3f}')

# Serialize the best model to a file using pickle.
pickle.dump(best_model, open('multinomial.pkl', 'wb'))











# Briefly explain the model output in the documentation. 



"""

The model achieves an accuracy of 99.7% on both the full dataset and the test set. This high accuracy indicates the model is 
very effective at predicting the loan_status based on the input features.
This accuracy level suggests that the model generalizes well to new, unseen data, reflecting the effectiveness and precision 
of the feature engineering, data preprocessing steps applied, and the robustness of the logistic regression model utilized.

"""








# Write about the benefits/impact of the solution - in what way does the business (client) benefit from the solution provided?




"""

Enhanced Decision-Making:
The model provides accurate predictions of loan status, allowing the business to make informed decisions about loan approvals, 
pricing strategies, and risk management. This can lead to more reliable and consistent lending practices.


Risk Management:
By identifying high-risk loans more effectively, the institution can better manage its portfolio risk. 
This results in reduced default rates and financial losses, thereby enhancing the overall financial stability of the institution.


Operational Efficiency:
Automating the classification of loan status through the model reduces the need for manual interventions 
and assessments, allowing loan officers to focus on more complex cases or strategic tasks. This increases the overall efficiency of the loan processing operation.


Regulatory Compliance:
With precise risk assessments and clear documentation of decision-making processes, the institution can better comply with strict regulatory 
requirements in the financial services industry, reducing the risk of sanctions or penalties.


Competitive Advantage:
By leveraging advanced data analytics and machine learning models, the business can gain a competitive edge in the market. 
Offering faster, more accurate assessments than competitors will attract more customers seeking efficiency and reliability.


Cost Reduction:
By minimizing default rates and optimizing resource allocations through precise predictions, the institution can reduce costs associated 
with collections, loss mitigation, and operational overheads.


"""



