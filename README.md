# 🤖 Predictive Modeling Using Machine Learning

## 📌 Overview
This project focuses on building machine learning models to predict customer purchase decisions based on various financial and behavioral features. It demonstrates the complete workflow of supervised learning, including data preprocessing, model training, and performance evaluation.

---

## 🎯 Objectives

- Build predictive models using machine learning algorithms  
- Train and test models for accuracy  
- Compare performance of different models  
- Visualize results using confusion matrices  

---

## 📂 Project Structure

```
project/
│
├── data/
│   ├── raw_data.csv
│   ├── cleaned_data.csv
│   └── ml_data.csv
│
├── src/
│   ├── cleaning.py
│   ├── visualization.py
│   └── model.py
│
├── output/
│   └── plots/
│
└── README.md
```
## ⚙️ Technologies Used

- Python  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## 📊 Dataset Description

The dataset includes the following features:

- Age  
- Income  
- Spending Score  
- Credit Score  
- Account Balance  
- Transactions  
- **Target:** Purchase Decision (0 = No, 1 = Yes)

---

## 🧹 Data Preprocessing

- Handled missing values using median imputation  
- Removed duplicate records  
- Treated outliers using IQR method  
- Prepared dataset for machine learning  

---

## 🤖 Machine Learning Models Used

- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  

---

## 📈 Model Evaluation

Models were evaluated using:

- Accuracy Score  
- Confusion Matrix  
- Classification Report  

---

## 📊 Output Visualizations

Confusion matrices for each model are generated and saved in:

output/plots/


---

## 🔍 Key Insights

- Random Forest generally performs better due to handling complex feature interactions  
- Higher income and account balance are strong indicators of purchase decisions  
- Credit score positively influences financial behavior  
- Ensemble models provide more stable predictions compared to single models  

---

## ▶️ How to Run the Project

### 1. Install dependencies

pip install -r requirements.txt

### 2. Run data cleaning

python src/cleaning.py

### 3. Run visualization

python src/visualization.py

### 4. Run machine learning model

python src/model.py

---

## 📁 Output

- Cleaned dataset  
- Visualizations  
- Confusion matrix plots  
- Model performance metrics  

---

## 🚀 Learning Outcomes

- Understanding supervised learning concepts  
- Model training and evaluation techniques  
- Comparing multiple machine learning models  
- Building end-to-end data science workflows  

---

## 👤 Author : Nithin Pratap A

Thiranex Internship - Data Science & Machine Learning Project
