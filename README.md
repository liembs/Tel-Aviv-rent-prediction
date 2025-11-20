# 🏙️ Tel Aviv Rent Prediction  
Machine Learning project for predicting apartment rental prices in Tel Aviv based on real-world property features, data cleaning, feature engineering, and model training.

---

## 📌 Project Overview  
This project predicts monthly rental prices in Tel Aviv using a Machine Learning pipeline that includes:

- Data preparation and cleaning  
- Feature selection & encoding  
- Model training (ElasticNet, Decision Tree)  
- Evaluation (MAE, R²)  
- Saving the trained model  
- Providing predictions via a **Flask API**

The goal is to demonstrate a complete end-to-end ML workflow:  
➡️ from raw data → to prediction API.

---

## 🗂 Project Structure


---

## 🧹 Data Preparation  
File: `scripts/assets_data_prep.py`

Includes:

- Renaming columns  
- Dropping irrelevant/faulty entries  
- Handling missing values  
- Outlier detection  
- Categorical encoding using `category_encoders`  
- Data normalization & preparation for training  

This step ensures a clean dataset and stable model performance.

---

## 🤖 Model Training  
File: `scripts/model_training.py`

Trained two regression models:

### 1️⃣ ElasticNet Regression  
- Good for structured data  
- Handles multicollinearity  
- Prevents overfitting  

### 2️⃣ DecisionTreeRegressor  
- Captures non-linear patterns  
- Easy to interpret  
- Useful for comparison  

Outputs include:

- **MAE (Mean Absolute Error)**
- **R² Score**
- Saved final model: `trained_model.pkl`

---

## 🔥 Prediction API  
File: `scripts/api.py`

A simple REST API built with Flask.

### ▶ Run the API locally:
```bash
python scripts/api.py


{
  "rooms": 3,
  "sqm": 68,
  "neighborhood": "Florentin"
}
{
  "predicted_rent": 5890
}
