# Heart-Disease-Prediction
Streamlit app for predicting heart disease risk using ML models
# 💓 Heart Disease Prediction using Machine Learning & Streamlit

This project is an **end-to-end machine learning application** that predicts the **risk of heart disease** based on a patient’s health indicators.  
It uses **Logistic Regression** and **Random Forest** models for reliable predictions and is deployed via an interactive **Streamlit web app**.

---

## 📘 Project Overview

Heart disease is one of the most common causes of death globally, and early detection can save lives.  
This project demonstrates how data science and machine learning can help **identify risk factors** and **support medical decisions**.

The project walks through the **entire ML workflow** — from **data collection** and **model training**, to **evaluation**, **explanation**, and **deployment** as a Streamlit app.

---

## 🧩 Project Workflow

### ** Data Collection**

The dataset used for this project is a variation of the **UCI Heart Disease Dataset**, containing clinical data such as:
- Age, sex, chest pain type, resting blood pressure, cholesterol, and more.
- Each record indicates whether the patient has heart disease (1) or not (0).

Data Source Example: [UCI Machine Learning Repository – Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)

---

### ** Data Cleaning and Preparation**

Using **Pandas**, I performed:
- Handling of missing or inconsistent data  
- Conversion of categorical features into numeric (e.g., Male/Female → 1/0)
- Feature scaling (for Logistic Regression)  
- Split into training and test sets (usually 80/20 split)
---
 Model Training
---
Two models were trained to ensure prediction reliability:

(a) Logistic Regression

Good for interpretable linear relationships.

Predicts probability of heart disease.

Coefficients help identify key risk factors.

(b) Random Forest Classifier

Non-linear ensemble model.

Handles complex feature interactions.

Provides feature importance ranking.
Both models were evaluated using:

Accuracy

Precision, Recall, F1-score

Confusion Matrix
---
 Model Comparison and Saving
---
After evaluation:

Both models performed well, but each had different strengths.

To reduce bias, I used both models together in the app.

✅ Decision Logic in App:
---
If both predict 1 → High Risk 💔

If one predicts 1 and the other 0 → Flag for Medical Review ⚠️

If both predict 0 - Low or Little Risk

Streamlit App Development
---

The app was built using Streamlit, a lightweight Python framework for ML app deployment.

User Inputs

Users can enter values like:

Age

Sex (Male/Female)

Chest Pain Type (0–3)

Resting Blood Pressure

Cholesterol

Fasting Blood Sugar

ECG Results

Max Heart Rate Achieved

Exercise-Induced Angina

ST Depression

Slope, Major Vessels, and Thalassemia type

Each input is explained clearly to help users understand the meaning of the numbers.

Prediction Logic

When the user clicks “Predict Heart Disease Risk”, the app:

Collects all input values.

Passes them into both models.

Displays the combined result in a user-friendly format.

---
Using Plotly, a gauge chart was added to visualize the patient’s risk level dynamically:

💚 Low Risk: 0–40%

🟡 Moderate Risk: 40–70%

❤️ High Risk: 70–100%

This provides an instant, intuitive understanding of the predicted result.

🖥️ How to Run the Project
---
Step 1: Clone the Repository

Step 2: Install Dependencies

Step 3: Run the App

---

🧑‍💻 Author
---
Toyeeb Timileyin Toye
📍 Data Analyst | Machine Learning Enthusiast

💼 Portfolio: (github.com/Tim-Tim100)

📧 Email: (toyeeb137@gmail.com)

🔗 LinkedIn: (https://www.linkedin.com/in/toyeeb-toye-805b0b247/)








	
