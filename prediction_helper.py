
import pandas as pd
import joblib


model_young = joblib.load("artifacts/model_young.joblib")
model_rest = joblib.load("artifacts/model_rest.joblib")
scaler_young = joblib.load("artifacts/scaler_young.joblib")
scaler_rest = joblib.load("artifacts/scaler_rest.joblib")

def handle_scaling(age, df):
    scaler_object = scaler_young if age <= 25 else scaler_rest

    # Access the correct keys
    cols_to_scale = scaler_object.get("cols_to_scale")
    if cols_to_scale is None:
        raise KeyError("The 'cols_to_scale' key is missing or incorrectly defined.")

    scaler = scaler_object["scaler"]

    df["income_level"] = None  # Add temporary column

    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df.drop("income_level", axis=1, inplace=True)

    return df


def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    # Split the medical history into potential two parts and convert to lowercase
    diseases = medical_history.lower().split(" & ")

    # Calculate the total risk score by summing the risk scores for each part
    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)  # Default to 0 if disease not found

    max_score = 14 # risk score for heart disease (8) + second max risk score (6) for diabetes or high blood pressure
    min_score = 0  # Since the minimum score is always 0

    # Normalize the total risk score
    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)

    return normalized_risk_score


def preprocess_input(input_dict):
    # Define the expected columns and initialize the DataFrame with zeros
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight', 'smoking_status_Occasional',
        'smoking_status_Regular', 'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}

    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    # Manually assign values for each categorical input based on input_dict
    for key, value in input_dict.items():
        if key == 'Gender' and value == 'Male':
            df['gender_Male'] = 1
        elif key == 'Region':
            if value == 'Northwest':
                df['region_Northwest'] = 1
            elif value == 'Southeast':
                df['region_Southeast'] = 1
            elif value == 'Southwest':
                df['region_Southwest'] = 1
        elif key == 'Marital Status' and value == 'Unmarried':
            df['marital_status_Unmarried'] = 1
        elif key == 'BMI Category':
            if value == 'Obesity':
                df['bmi_category_Obesity'] = 1
            elif value == 'Overweight':
                df['bmi_category_Overweight'] = 1
            elif value == 'Underweight':
                df['bmi_category_Underweight'] = 1
        elif key == 'Smoking Status':
            if value == 'Occasional':
                df['smoking_status_Occasional'] = 1
            elif value == 'Regular':
                df['smoking_status_Regular'] = 1
        elif key == 'Employment Status':
            if value == 'Salaried':
                df['employment_status_Salaried'] = 1
            elif value == 'Self-Employed':
                df['employment_status_Self-Employed'] = 1
        elif key == 'Insurance Plan':
            df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
        elif key == 'Age':
            df['age'] = value
        elif key == 'Number of Dependants':
            df['number_of_dependants'] = value
        elif key == 'Income in Lakhs':
            df['income_lakhs'] = value
        elif key == "Genetical Risk":
            df['genetical_risk'] = value

    # Remove any unwanted columns like 'unnamed:_0' or other extra columns
    df = df.loc[:, ~df.columns.str.contains('unnamed:_0', case=False)]
    
    # Add the 'unnamed:_0' column explicitly (with zeros or another placeholder value)
    df['unnamed:_0'] = 0

    # Calculate the normalized risk score
    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])

    # Scale the dataframe using the appropriate scaler
    df = handle_scaling(input_dict['Age'], df)

    if input_dict['Age'] <= 25:
        model_features = model_young.get_booster().feature_names
        df = df[model_features] # Reorder columns to match expected features
    else:
        model_features = model_rest.get_booster().feature_names
        df = df[model_features] # Reorder columns to match expected features

    return df

input_dict = {'Age': 25, 'Number of Dependants': 0, 'Income in Lakhs': 0, 'Genetical Risk': 0, 'Insurance Plan': 'Bronze', 'Employment Status': 'Salaried', 'Gender': 'Male', 'Marital Status': 'Unmarried', 'BMI Category': 'Normal', 'Smoking Status': 'No Smoking', 'Region': 'Northwest', 'Medical History': 'No Disease'}

def predict(input_dict):
    input_df = preprocess_input(input_dict)
    if input_dict['Age'] <= 25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)
    return int(prediction[0])


print(predict(input_dict))


















# import pandas as pd
# import numpy as np
# from joblib import load

# model_rest = load(r"app\artifacts\model_rest.joblib")
# model_young = load(r"app\artifacts\model_young.joblib")
# scaler_rest = load(r"app/artifacts/scaler_rest.joblib")
# scaler_young = load(r"app/artifacts/scaler_young.joblib")

# def calculate_risk_score(medical_history):

#     risk_score = {
#     "diabetes" : 6,
#     "heart disease" : 8,
#     "high blood pressure" : 6,
#     "thyroid" : 5,
#     "no disease" : 0,
#     "none" : 0
#     }

#     # split the model into two parts and convert to lowercase
#     diseases = medical_history.lower().split(" & ")

#     # calculate the total risk by summing the risk score for each part
#     total_risk_score = sum(risk_score.get(disease,0) for disease in diseases) #default to 0 if disease not found

#     max_score = 14
#     min_score = 0

#     normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)

#     return normalized_risk_score


# def preprocess_input(input_dict):
#     expected_cols = ['age', 'number_of_dependants', 'income_lakhs', 'insurance_plan',
#        'genetical_risk', 'normalized_risk_score', 'gender_Male',
#        'region_Northwest', 'region_Southeast', 'region_Southwest',
#        'marital_status_Unmarried', 'bmi_category_Obesity',
#        'bmi_category_Overweight', 'bmi_category_Underweight',
#        'smoking_status_Occasional', 'smoking_status_Regular',
#        'employment_status_Salaried', 'employment_status_Self-Employed']


#     insurance_plan_encoding = {'Bronze' : 1, 'Silver' : 2, 'Gold' : 3}

#     df = pd.DataFrame(0, columns=expected_cols, index=[0])

#     for key, value in input_dict.items():
#         if key == 'Gender' and value == 'Male':
#             df['gender_Male'] = 1
#         elif key == 'Region':
#             if value == 'Northwest':
#                 df['region_Northwest'] = 1
#             elif value == 'Southeast':
#                 df['region_Southeast'] = 1
#             elif value == 'Southwest':
#                 df['region_Southwest'] = 1
#         elif key == 'Marital Status' and value == 'Unmarried':
#             df['marital_status_Unmarried'] = 1
#         elif key == 'BMI Category':
#             if value == 'Obesity':
#                 df['bmi_category_Obesity'] = 1
#             elif value == 'Overweight':
#                 df['bmi_category_Overweight'] = 1
#             elif value == 'Underweight':
#                 df['bmi_category_Underweight'] = 1
#         elif key == 'Smoking Status':
#             if value == 'Occasional':
#                 df['smoking_status_Occasional'] = 1
#             elif value == 'Regular':
#                 df['smoking_status_Regular'] = 1
#         elif key == 'Employment Status':
#             if value == 'Salaried':
#                 df['employment_status_Salaried'] = 1
#             elif value == 'Self-Employed':
#                 df['employment_status_Self-Employed'] = 1
#         elif key == 'Insurance Plan':  # Correct key usage with case sensitivity
#             df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
#         elif key == 'Age':  # Correct key usage with case sensitivity
#             df['age'] = value
#         elif key == 'Number of Dependants':  # Correct key usage with case sensitivity
#             df['number_of_dependants'] = value
#         elif key == 'Income in Lakhs':  # Correct key usage with case sensitivity
#             df['income_lakhs'] = value
#         elif key == "Genetical Risk":
#             df['genetical_risk'] = value

        
#     df["normalized_risk"] = calculate_risk_score(input_dict["Medical History"])

#     df = handle_scaling(input_dict["Age"], df)
#     return df

# def handle_scaling(age, df):
#     if age <= 25:
#         scaler_object = scaler_young
#     else:
#         scaler_object = scaler_rest

#     cols_to_scale = scaler_object["cols_to_scale"]
#     scaler = scaler_object["scaler"]

#     df["income_level"] = None
#     df[cols_to_scale] = scaler.transform(df[cols_to_scale])
#     df.drop("income_level", axis = 1, inplace = True)

#     return df

# def predict(input_dict):
#     input_df = preprocess_input(input_dict)

#     if input_dict["Age"] <= 25:
#         prediction = model_young.predict(input_df)
#     else:
#         prediction = model_rest.predict(input_df)

#     return int(prediction)

# print(predict())