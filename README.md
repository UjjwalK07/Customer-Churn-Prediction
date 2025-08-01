# Customer Churn Prediction using Deep Learning 📉🧠

This project predicts customer churn for a telecom company using an **Artificial Neural Network (ANN)**. It is a classic **binary classification** problem where the goal is to identify whether a customer will **leave (churn)** or **stay** based on their profile and behavior patterns.

---

## 🔍 Problem Statement

Telecom companies often face challenges in retaining customers. Churn prediction enables businesses to **identify high-risk customers** and take **proactive actions** to improve retention. In this project, we use a **deep learning model** to predict churn based on various customer attributes.

---

## ✅ Features Used

- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Score
- Active Membership
- Estimated Salary
- Has Credit Card

---

## 🧠 Model Overview

- Built using **TensorFlow** and **Keras**
- Model Type: **Artificial Neural Network (ANN)**
- Optimizer: **Adam**
- Loss Function: **Binary Crossentropy**
- Evaluation Metric: **Accuracy**

---

## 🧪 Steps Performed

1. Data Cleaning & Preprocessing
2. Label Encoding for Categorical Variables
3. Feature Scaling using StandardScaler
4. Splitting the Dataset into Training and Testing
5. Building and Compiling the ANN
6. Training the Model
7. Evaluating the Performance

---

## 📁 Folder Structure

```
Customer-Churn-Prediction/
│
├── Churn_Modelling.csv            # Dataset
├── experiment.ipynb               # Jupyter Notebook with model code
├── README.md                      # Project Overview
├── .gitignore                     # Files and folders ignored by Git
└── venv/                          # Virtual environment (should be ignored)
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/saiengineerss/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate       # On Windows
# source venv/bin/activate  # On Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open and run the notebook

# 5. Open app.py and run the file
streamlit run app.py
```

---

## 📊 Model Performance

- Achieved **high accuracy** on the test dataset
- Evaluation done using:
  - Accuracy Score ~85%
  - Confusion Matrix
  - Classification Report

---

## 🙌 Acknowledgements

- Dataset inspired by publicly available Kaggle datasets
- Built using:
  - Python
  - Pandas & NumPy
  - Scikit-learn
  - TensorFlow & Keras
  - Streamlit



## 🛑 Note

- Make sure the `venv/` folder is **excluded** from the repository using `.gitignore`
- This project is for **educational purposes**

