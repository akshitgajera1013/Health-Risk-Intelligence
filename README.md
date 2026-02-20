# 🏥 Health Risk Intelligence Platform
AI-Based Lifestyle Disease Risk Assessment System  
Built using Tuned Decision Tree (RandomizedSearchCV)

Deployment Link :- https://health-risk-intelligence-model.streamlit.app/

## 🚀 Project Overview
This project predicts potential health disease risk based on lifestyle, physiological, and behavioral factors using a hyperparameter-optimized Decision Tree Classifier.

It provides:
- 🎯 Risk Prediction
- 📊 Probability Distribution
- 📈 Interactive Partial Dependence Analysis
- 🌳 Decision Tree Visualization
- 📌 Feature Importance Analysis
- ⚙ Best Hyperparameter Insights
- 🌡 Risk Scoring Meter

## 🧠 Machine Learning Details
| Item | Value |
| Algorithm | Decision Tree Classifier |
| Hyperparameter Tuning | RandomizedSearchCV |
| Accuracy | ~75% |
| Features | 14 |
| Output | Binary Risk Classification |

## 📊 Features Used
- Age
- Gender
- BMI
- Daily Steps
- Sleep Hours
- Water Intake
- Calories Consumed
- Smoker
- Alcohol
- Resting Heart Rate
- Systolic BP
- Diastolic BP
- Cholesterol
- Family History


## 📈 Application Sections

### 1️⃣ Prediction Tab
- User enters lifestyle details
- Model predicts risk
- Confidence score displayed
- Risk Gauge visualization

### 2️⃣ Analytics Tab
- Probability distribution chart
- Interactive Partial Dependence Plot

### 3️⃣ Model Insights Tab
- Full Decision Tree visualization
- Feature importance ranking
- Best hyperparameters display

## 🛠 Tech Stack
- Python
- Streamlit
- Scikit-Learn
- Plotly
- Matplotlib
- Pandas
- NumPy

## ▶️ How To Run Locally
git clone https://github.com/akshitgajera1013/Health-Risk-Intelligence.git

cd Health-Risk-Intelligence   

pip install -r requirements.txt

streamlit run app.py
