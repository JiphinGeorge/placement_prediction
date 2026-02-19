🎓 Student Placement Prediction System

A Machine Learning web application that predicts whether a student will be placed based on academic performance, work experience, and employability metrics. This project uses the Kaggle Campus Recruitment Dataset and an optimized XGBoost model with hyperparameter tuning and feature engineering.

📌 Project Overview

This system analyzes student academic and employability data to predict placement status using a trained machine learning model. The model is deployed through an interactive Streamlit web application.

🚀 Features

Machine Learning prediction using XGBoost

89.30% accuracy achieved using GridSearchCV

Feature engineering and feature selection

Interactive web interface using Streamlit

Real-time placement prediction

Clean and professional dashboard

🧠 Machine Learning Details

Algorithm used:

XGBoost Classifier

Techniques applied:

Label Encoding

Feature Engineering

Feature Selection

GridSearchCV Hyperparameter Tuning

Stratified Cross Validation

Accuracy achieved:
89.30%

📂 Project Structure
placement_prediction/
│
├── data/
│   └── placement.csv
│
├── model/
│   ├── placement_model.pkl
│   └── label_encoders.pkl
│
├── app.py
├── train_model.py
├── README.md

⚙️ Installation

Install required libraries:

pip install pandas numpy scikit-learn xgboost streamlit joblib

▶️ How to Run the Project

Step 1: Train the model (optional, already trained model is included)

python train_model.py


Step 2: Run the Streamlit web application

python -m streamlit run app.py


Step 3: Open browser and go to:

http://localhost:8501

📊 Input Features Used

Gender

SSC Percentage

HSC Percentage

Degree Percentage

MBA Percentage

Work Experience

Employability Test Percentage

MBA Specialisation

Academic Average (engineered feature)

Employability Score (engineered feature)

🖥️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

XGBoost

Streamlit

Joblib

📈 Dataset

Dataset used:
Kaggle Campus Recruitment Dataset

Contains student academic and placement records.

🎯 Use Cases

Predict student placement chances

Academic performance analysis

Educational analytics projects

Machine Learning portfolio project

👨‍💻 Author

Jiphin George
MCA Student
Machine Learning Enthusiast

📌 Future Improvements

Deploy application online

Add database integration

Improve UI design

Add visualization dashboard

⭐ How to Use

Enter student details in sidebar

Click Predict Placement

View placement prediction and probability

✅ Status

Project completed and ready for deployment.