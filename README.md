# Diabetes Prediction System

This project is a Machine Learning based web application that predicts whether a patient is likely to have diabetes using medical diagnostic measurements.

The project uses the K-Nearest Neighbors (KNN) algorithm along with SMOTE balancing and a Streamlit web interface.

---

# Features

- Diabetes Prediction using Machine Learning
- Data Cleaning and Preprocessing
- Feature Scaling using StandardScaler
- SMOTE for handling imbalanced data
- KNN Classification Model
- Interactive Streamlit Web Application
- User-friendly interface

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Imbalanced-Learn
- Streamlit
- Joblib
- Matplotlib
- Seaborn

---

# Project Structure

Diabetes-Project/

├── app.py  
├── train_model.py  
├── diabetes-data.csv  
├── diabetes_model.pkl  
├── scaler.pkl  
├── requirements.txt  
└── README.md  

---

# Dataset Information

The dataset contains medical information such as:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

Target Variable:

- Outcome
  - 0 = Non-Diabetic
  - 1 = Diabetic

---

# Machine Learning Workflow

1. Load Dataset
2. Data Cleaning
3. Replace Invalid Zero Values
4. Feature Scaling
5. Train-Test Split
6. Apply SMOTE
7. Train KNN Model
8. Evaluate Accuracy
9. Save Model
10. Deploy Streamlit App

---

# Installation

Install required libraries:

```bash
pip install -r requirements.txt
