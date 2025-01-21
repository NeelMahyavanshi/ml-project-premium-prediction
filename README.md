# **Health Insurance Premium Predictor**

This project predicts health insurance premiums based on user inputs like age, income, BMI category, smoking status, and medical history. It uses machine learning models trained on historical data to provide accurate and reliable predictions.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Details](#model-details)
6. [Technologies Used](#technologies-used)
7. [Project Structure](#project-structure)
8. [License](#license)

---

## **Project Overview**
The Health Insurance Premium Predictor is an interactive web application built with Streamlit. It leverages machine learning models to estimate premiums based on various factors:
- **Age**
- **Number of Dependents**
- **Income**
- **Medical History**
- **Smoking and BMI Status**
- **Region**

The app is designed to provide users with insights into how their personal and health-related attributes impact insurance costs.

---

## **Features**
- **Interactive Web Interface**: Built with Streamlit for an easy-to-use experience.
- **Dynamic Inputs**: Support for categorical and numerical features.
- **Accurate Predictions**: Powered by XGBoost models for young and older adults.
- **Feature Scaling**: Custom preprocessing pipelines for input normalization.

---

## **Installation**

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/ml-project-premium-prediction.git
   cd ml-project-premium-prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # For Linux/Mac
   env\Scripts\activate     # For Windows
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## **Usage**
1. Open the app in your browser (Streamlit will provide a local URL).
2. Enter the required inputs in the form (e.g., Age, Income, Region).
3. Click the **Predict** button.
4. View the predicted insurance premium.

---

## **Model Details**
- **Models Used**:
  - **XGBoost**: Separate models for users aged ≤25 and >25.
- **Preprocessing**:
  - Categorical encoding, normalization of numerical features, and feature engineering (e.g., risk score calculation).

---

## **Technologies Used**
- **Languages**: Python
- **Libraries**:
  - Streamlit
  - Pandas
  - Numpy
  - Scikit-learn
  - XGBoost
  - Joblib

---

## **Project Structure**
```
ml-project-premium-prediction/
│
├── artifacts/
│   ├── model_young.joblib
│   ├── model_rest.joblib
│   ├── scaler_young.joblib
│   └── scaler_rest.joblib
│
├── app.py                # Main Streamlit application
├── prediction_helper.py  # Functions for prediction and preprocessing
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## **License**
This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

---

