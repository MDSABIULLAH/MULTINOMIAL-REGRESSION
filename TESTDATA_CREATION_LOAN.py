import pandas as pd

# Define the data dictionary
data = {
    "id": [1077501, 1077430, 1077175, 1076863, 1075358, 1070078, 1069908, 1064687, 1069866, 1069057, 1069759, 1065775, 1069971, 1062474],
    "member_id": [1296599, 1314167, 1313524, 1277178, 1311748, 1305201, 1305008, 1298717, 1304956, 1303503, 1304871, 1299699, 1304884, 1294539],
    "loan_amnt": [5000, 2500, 2400, 10000, 3000, 6500, 12000, 9000, 3000, 10000, 1000, 10000, 3600, 6000],
    "funded_amnt": [5000, 2500, 2400, 10000, 3000, 6500, 12000, 9000, 3000, 10000, 1000, 10000, 3600, 6000],
    "funded_amnt_inv": [4975, 2500, 2400, 10000, 3000, 6500, 12000, 9000, 3000, 10000, 1000, 10000, 3600, 6000],
    "term": ["36 months", "60 months", "36 months", "36 months", "60 months", "60 months", "36 months", "36 months", "36 months", "36 months", "36 months", "36 months", "36 months", "36 months"],
    "int_rate": ["10.65%", "15.27%", "15.96%", "13.49%", "12.69%", "14.65%", "12.69%", "13.49%", "9.91%", "10.65%", "16.29%", "15.27%", "6.03%", "11.71%"],
    "installment": [162.87, 59.83, 84.33, 339.31, 67.79, 153.45, 402.54, 305.38, 96.68, 325.74, 35.31, 347.98, 109.57, 198.46],
    "grade": ["B", "C", "C", "C", "B", "C", "B", "C", "B", "B", "D", "C", "A", "B"],
    "sub_grade": ["B2", "C4", "C5", "C1", "B5", "C3", "B5", "C1", "B1", "B2", "D1", "C4", "A1", "B3"],
    "home_ownership": ["RENT", "OTHER", "RENT", "OTHER", "RENT", "OWN", "OWN", "NONE", "RENT", "RENT", "NONE", "RENT", "MORTGAGE", "MORTGAGE"],
    "annual_inc": [24000, 30000, 12252, 49200, 80000, 72000, 75000, 30000, 15000, 100000, 28000, 42000, 110000, 84000],
    "verification_status": ["Verified", "Source Verified", "Not Verified", "Source Verified", "Source Verified", "Not Verified", "Source Verified", "Source Verified", "Not Verified", "Source Verified", "Not Verified", "Not Verified", "Source Verified", "Verified"],
    "issue_d": ["11-Dec"] * 14,
    "pymnt_plan": ["n"] * 14,
    "url": [
        "https://lendingclub.com/browse/loanDetail.action?loan_id=1077501",
        "https://lendingclub.com/browse/loanDetail.action?loan_id=1077430",
        "https://lendingclub.com/browse/loanDetail.action?loan_id=1077175",
        "https://lendingclub.com/browse/loanDetail.action?loan_id=1076863",
        "https://lendingclub.com/browse/loanDetail.action?loan_id=1075358",
        "https://lendingclub.com/browse/loanDetail.action?loan_id=1070078",
        "https://lendingclub.com/browse/loanDetail.action?loan_id=1069908",
        "https://lendingclub.com/browse/loanDetail.action?loan_id=1064687",
        "https://lendingclub.com/browse/loanDetail.action?loan_id=1069866",
        "https://lendingclub.com/browse/loanDetail.action?loan_id=1069057",
        "https://lendingclub.com/browse/loanDetail.action?loan_id=1069759",
        "https://lendingclub.com/browse/loanDetail.action?loan_id=1065775",
        "https://lendingclub.com/browse/loanDetail.action?loan_id=1069971",
        "https://lendingclub.com/browse/loanDetail.action?loan_id=1062474",
    ],
    "purpose": ["credit_card", "car", "small_business", "moving", "other", "vacation", "renewable_energy", "wedding", "educational", "house", "debt_consolidation", "home_improvement", "major_purchase", "medical"],
    "zip_code": ["860xx", "309xx", "606xx", "917xx", "972xx", "853xx", "913xx", "245xx", "606xx", "951xx", "641xx", "921xx", "067xx", "890xx"],
    "addr_state": ["AZ", "GA", "IL", "CA", "OR", "AZ", "CA", "VA", "IL", "CA", "MO", "CA", "CT", "UT"],
    "dti": [27.65, 1, 8.72, 20, 17.94, 16.12, 10.78, 10.08, 12.56, 7.06, 20.31, 18.6, 10.52, 18.44],
    "delinq_2yrs": [0] * 14,
    "earliest_cr_line": ["Jan-85", "Apr-99", "Nov-01", "Feb-96", "Jan-96", "Jan-98", "Oct-89", "Apr-04", "Jul-03", "May-91", "Sep-07", "Oct-98", "Aug-93", "Oct-03"],
    "inq_last_6mths": [1, 5, 2, 1, 0, 2, 0, 1, 2, 2, 1, 2, 0, 0],
    "open_acc": [3, 3, 2, 10, 15, 14, 12, 4, 11, 14, 11, 14, 20, 4],
    "pub_rec": [0] * 14,
    "revol_bal": [13648, 1687, 2956, 5598, 27783, 4032, 23336, 10452, 7323, 11997, 6524, 24043, 22836, 0],
    "total_acc": [9, 4, 10, 37, 38, 23, 34, 9, 11, 29, 23, 28, 42, 14],
    "initial_list_status": ["f"] * 14,
    "out_prncp": [0] * 14,
    "out_prncp_inv": [0] * 14,
    "total_pymnt": [5863.16, 1008.71, 3005.67, 12231.89, 3513.33, 7678.02, 13947.99, 2270.7, 3480.27, 7471.99, 1270.72, 12527.15, 3785.27, 7167.07],
    "total_pymnt_inv": [5833.84, 1008.71, 3005.67, 12231.89, 3513.33, 7678.02, 13947.99, 2270.7, 3480.27, 7471.99, 1270.72, 12527.15, 3785.27, 7167.07],
    "total_rec_prncp": [5000, 456.46, 2400, 10000, 2475.94, 6500, 12000, 1256.14, 3000, 5433.47, 1000, 10000, 3600, 6000],
    "total_rec_int": [863.16, 435.17, 605.67, 2214.92, 1037.39, 1178.02, 1947.99, 570.26, 480.27, 1393.42, 270.72, 2527.15, 185.27, 1152.07],
    "total_rec_late_fee": [0] * 14,
    "recoveries": [0, 117.08, 0, 16.97, 0, 0, 0, 444.3, 0, 645.1, 0, 0, 0, 0],
    "collection_recovery_fee": [0, 1.11, 0, 0, 0, 0, 0, 4.16, 0, 6.31, 0, 0, 0, 0],
    "last_pymnt_amnt": [171.62, 119.66, 649.91, 357.48, 67.79, 1655.54, 6315.3, 305.38, 102.43, 325.74, 36.32, 370.46, 583.45, 16.98],
    "policy_code": [1] * 14,
    "application_type": ["INDIVIDUAL"] * 14,
    "acc_now_delinq": [0] * 14,
    "delinq_amnt": [0] * 14
}

# Create a DataFrame using the data dictionary
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('loan_data_test.csv', index=False)

print("Data has been successfully written to 'loan_data_test.csv'.")


        


        
       


   
