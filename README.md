# AI-DISEASE-PREDICTOR 🩺
Hands-on ML project covering preprocessing, model training, evaluation, optimization &amp; healthcare prediction.

## 📌 Dataset Overview

We use healthcare datasets (e.g., **Heart Disease, Diabetes**) with features such as patient vitals, medical history, and lifestyle factors.  

### Example: Heart Disease Dataset  

| Feature             | Description                                     | Type        |
|---------------------|-------------------------------------------------|-------------|
| Age                 | Age of the patient                              | Numeric     |
| Sex                 | Gender (1 = Male, 0 = Female)                   | Categorical |
| ChestPainType       | Type of chest pain                              | Categorical |
| RestingBP           | Resting blood pressure (mm Hg)                  | Numeric     |
| Cholesterol         | Serum cholesterol (mg/dl)                       | Numeric     |
| FastingBS           | Fasting blood sugar > 120 mg/dl (1 = True, 0 = False) | Binary |
| MaxHR               | Maximum heart rate achieved                     | Numeric     |
| ExerciseAngina      | Exercise induced angina (1 = Yes, 0 = No)       | Binary      |
| Oldpeak             | ST depression induced by exercise               | Numeric     |
| Target              | Disease (1 = Present, 0 = Absent)               | Binary      |

## 📊 Data Analysis (Exploratory Insights)

| Metric / Feature             | Observation Example (Dataset)        |
|-------------------------------|--------------------------------------|
| Average Age                   | 54.6 years                          |
| Male : Female Ratio           | 68% : 32%                           |
| Average Cholesterol           | 245 mg/dl                           |
| Max Heart Rate Achieved       | 150 bpm                             |
| % Patients with Heart Disease | 46%                                 |

## 🧠 Model Training & Evaluation

We tested multiple ML models on the dataset:  

| Model                   | Accuracy | Precision | Recall | F1-Score |
|--------------------------|----------|-----------|--------|----------|
| Logistic Regression      | 82%      | 0.81      | 0.78   | 0.79     |
| Random Forest Classifier | 87%      | 0.85      | 0.84   | 0.84     |
| SVM (RBF Kernel)         | 85%      | 0.83      | 0.81   | 0.82     |
| K-Nearest Neighbors      | 80%      | 0.78      | 0.76   | 0.77     |

👉 **Best Performing Model:** Random Forest Classifier (Accuracy: 87%)  

## 📚 5-Day Breakdown

| Day | Topics Covered |
|-----|----------------|
| 1   | ML Basics, Data Preprocessing |
| 2   | Model Training & Evaluation |
| 3   | Advanced Models & Feature Engineering |
| 4   | Model Optimization & Hyperparameter Tuning |
| 5   | Live Prediction Demo & Portfolio Showcase |

## 🛠️ Tech Stack
- **Language:** Python 3  
- **Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn  
- **Platform:** Google Colab / Jupyter Notebook  

cd Disease-Predictor-Bootcamp
pip install -r requirements.txt
