# 🫀 Heart Disease Predictor  
An AI-powered ML project for predicting the likelihood of heart disease based on patient health parameters.  
This project demonstrates end-to-end machine learning including **data preprocessing, feature engineering, model training, evaluation, and deployment using Streamlit**.  

---

## 📌 Dataset Overview  

We used the **UCI Heart Disease dataset** with patient health records.  
The dataset contains **13 clinical features** that are transformed into **30 model-ready features** using preprocessing and one-hot encoding.  

### Example Features  

| Feature             | Description                                     | Type        |
|---------------------|-------------------------------------------------|-------------|
| Age                 | Age of the patient                              | Numeric     |
| Sex                 | Gender (1 = Male, 0 = Female)                   | Categorical |
| Chest Pain Type     | Type of chest pain (0–3)                        | Categorical |
| RestingBP           | Resting blood pressure (mm Hg)                  | Numeric     |
| Cholesterol         | Serum cholesterol (mg/dl)                       | Numeric     |
| FastingBS           | Fasting blood sugar > 120 mg/dl (1 = True, 0 = False) | Binary |
| Rest ECG            | Resting electrocardiographic results (0–2)      | Categorical |
| MaxHR               | Maximum heart rate achieved                     | Numeric     |
| Exercise Angina     | Exercise induced angina (1 = Yes, 0 = No)       | Binary      |
| Oldpeak             | ST depression induced by exercise               | Numeric     |
| Slope               | Slope of the peak exercise ST segment (0–2)     | Categorical |
| Major Vessels (ca)  | Number of major vessels colored by fluoroscopy  | Numeric     |
| Thalassemia (thal)  | Blood disorder type (0–2)                       | Categorical |

---

## 📊 Data Insights  

| Metric / Feature             | Observation Example (Dataset)        |
|-------------------------------|--------------------------------------|
| Average Age                   | ~54 years                           |
| Male : Female Ratio           | 68% : 32%                           |
| Average Cholesterol           | ~245 mg/dl                          |
| Max Heart Rate Achieved       | ~150 bpm                            |
| % Patients with Heart Disease | ~46%                                |

---

## 🧠 Model Training & Evaluation  

We trained and compared ML models on the dataset:  

| Model                   | Accuracy | Precision | Recall | F1-Score |
|--------------------------|----------|-----------|--------|----------|
| Logistic Regression      | 82%      | 0.81      | 0.78   | 0.79     |
| Random Forest Classifier | **87%**  | 0.85      | 0.84   | 0.84     |

👉 **Best Performing Model:** Random Forest Classifier (**Accuracy: 87%**)  

---

## 🛠️ Tech Stack  

- **Language:** Python 3  
- **Libraries:** scikit-learn, pandas, numpy, joblib, matplotlib, seaborn, streamlit  
- **Platform:** Google Colab  / Streamlit Cloud  

---

Developer : DEBASMITA KARMAKAR
Email : dbsmita06@gmail.com
