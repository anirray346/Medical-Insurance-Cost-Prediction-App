# Medical-Insurance-Cost-Prediction-App using Machine Learning 

## 🚀 Project Summary
This project builds an end-to-end **machine learning regression pipeline** to predict medical insurance costs based on individual attributes such as age, BMI, smoking status, and region. Multiple models are trained, evaluated, and optimized to identify a **high-performing, well-generalized solution** suitable for real-world deployment.

---

## 🎯 Business Problem
Accurately estimating medical insurance costs helps insurers:
- Assess financial risk
- Design fair premium policies
- Improve customer segmentation

This project demonstrates how **data-driven modeling** can support such decisions.

---

## 📊 Dataset Overview
The dataset includes demographic and health-related features:

| Feature | Description |
|------|-------------|
| age | Age of beneficiary |
| sex | Gender |
| bmi | Body Mass Index |
| children | Number of dependents |
| smoker | Smoking status |
| region | Residential region |
| charges | Medical insurance cost (target) |

---

## 🔍 Exploratory Data Analysis
Key insights derived from EDA:
- **Smoking status** is the strongest predictor of insurance charges.
- Medical costs increase with **age** and **BMI**.
- High-charge outliers represent genuine medical cases and were retained.
- Feature relationships were visualized using histograms, boxplots, and correlation heatmaps.

---

## 🧹 Data Preparation & Feature Engineering
- One-hot encoding applied to categorical variables
- Feature scaling using **StandardScaler** for fair model contribution
- Skewness analyzed and retained due to robustness of tree-based models
- Outliers preserved to maintain real-world data integrity

---

## 🤖 Models & Algorithms
The following regression models were implemented and compared:

- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- Support Vector Regressor (SVR)  
- K-Nearest Neighbors (KNN)  
- Ensemble Learning (Voting Regressor)

---

## 📈 Model Evaluation
Models were evaluated using:
- MAE, MSE, RMSE
- R² and Adjusted R²
- Train vs Test performance to detect overfitting

📌 **Key Insight:**  
Tree-based models outperformed linear and kernel-based models for this dataset.

---

## 🛠 Model Optimization
- **GridSearchCV** was used to tune Random Forest hyperparameters
- Tuning reduced overfitting and improved test performance
- The tuned Random Forest achieved the **best balance of accuracy and generalization**

---

## 🏆 Best Model
**Tuned Random Forest Regressor**
- Highest Test R²
- Lowest Test RMSE
- Minimal overfitting

The trained model and scaler were saved using `joblib`, making the solution deployment-ready.

---

## 💡 What This Project Demonstrates
✔ End-to-end ML workflow  
✔ Strong EDA & feature reasoning  
✔ Model comparison & selection  
✔ Overfitting awareness  
✔ Hyperparameter optimization  
✔ Production-ready mindset  

---

## 🔮 Future Enhancements
- Build REST API for predictions
- Integrate advanced ensemble models (XGBoost, LightGBM)

---

## 🧑‍💻 Author
**Anirban Ray**  
Aspiring Data Scientist | Machine Learning Enthusiast  

📫 *Feel free to connect for collaboration or opportunities.*

---

## 📜 License
For educational and portfolio use.
