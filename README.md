# Income Prediction – Supervised Learning Project  

**Contributors:**  
- **Aymeric Mouttapa**  
- **Leon Pejic**

---

## 1. Introduction

[Fill in]
---

## 2. Dataset Description

The project uses the **Adult Census Income** dataset containing demographic, educational and professional attributes.

### Key variables:

**Categorical features**  
- workclass  
- education  
- marital-status  
- occupation  
- relationship  
- race  
- sex  
- native-country  

**Numeric features**  
- age  
- education-num  
- capital-gain  
- capital-loss  
- hours-per-week  
- estimation_carriere_age  
- is_married  
- maried_education_experience  

**Engineered features** (in `income_boosted.csv`):  
- gain_per_hour  
- loss_per_hour  
- edu_age_ratio  
- log_gain  
- age bins  
- estimation_carriere_age  
- maried_education_experience = education-num × age × is_married  

**Target variable:**  
`income` → 0 = <=50K, 1 = >50K

### Dataset files included:

| File | Description |
|------|-------------|
| `income_1996.csv` | Original dataset |
| `income_cleaned.csv` | Cleaned dataset |
| `income_boosted.csv` | Dataset with engineered features |

---

## 3. Project Steps

### 3.1. Exploratory Data Analysis  
We analyzed in **eda.ipynb** :

- Feature distributions  
- Relationship between income and education, occupation, age, hours worked  
- Correlation heatmap  
- Outliers  
- Key predictors  

After this we got a csv called *income_cleaned.csv*.

### 3.2. Data Cleaning & Feature Engineering  
We built additional features including in **feature_boost_test.ipynb**:

- Ratios (`gain_per_hour`, `loss_per_hour`)  
- Log-transformed capital gains  
- Age bins  
- Education/age interactions  
- Marital status transformations  
- `maried_education_experience`  

After this we got a csv called *income_boosted.csv*.

### 3.3.1 Linear Regression

We briefly tested a Linear Regression model as a baseline, even though the target is categorical. 
As expected, its performance was poor, and it was discarded early in favor of more appropriate classification models.

### 3.3.2 Baseline Model: RandomForest  
- Preprocessing: StandardScaler + OneHotEncoder  
- Accuracy ≈ 0.84  
- F1 ≈ 0.84  
- ROC-AUC ≈ 0.89  

### 3.4. Experiment Tracking (MLflow)

We tested:

- RandomForest  
- XGBoost  
- LightGBM  
- CatBoost  
- Stacking models  
- Feature engineering  
- Hyperparameter tuning  

Each experiment was tracked with MLflow (params + metrics + artifacts).

### 3.5. Best Model Selection

| Rank | Model | Dataset | F1 | ROC-AUC |
|------|--------|-----------|---------|---------|
| 1 | **LightGBM** | boosted | **0.8627** | **0.9258** |
| 2 | CatBoost | boosted | 0.8622 | 0.9252 |
| 3 | XGBoost | boosted | 0.8622 | 0.9257 |

→ **Final choice: LightGBM** (best performance + speed + deployability)
LightGBM was chosen as the final model because it provided the best combination of F1-score, ROC-AUC, and inference speed while remaining stable during training.

### 3.6. Final Training Pipeline (`main.py`)

`main.py`:

- Loads dataset  
- Applies preprocessing  
- Trains LightGBM  
- Computes metrics  
- Logs everything in MLflow  
- Saves the trained pipeline → `final_lightgbm_pipeline.joblib`  

### 3.7. Streamlit for live BUC

**Streamlit (`app_streamlit.py`)**  
- Dropdowns for categorical features  
- Numeric inputs  
- Automatic computation of:  
  - `education-num` from `education`  
  - `maried_education_experience`  
- Returns predicted income class and probability  
- Designed for marketing persona exploration  

---

## 4. How to Reproduce the Project

### 4.1. Python Version  
Recommended: **Python 3.10 or 3.11**

### 4.2. Clone the repository
```bash
git clone <your_repo_url>
cd supervised-learning
```

### 4.3. Create a virtual environment

#### Windows (PowerShell):

```bash
python -m venv venv
venv\Scripts\activate
```

#### MacOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4.4. Install dependencies

```bash
pip install -r requirements.txt
```

### 4.5. Add datasets to the project root

Place the following CSV files in the root of the repository:
income_1996.csv
income_cleaned.csv
income_boosted.csv

### 4.6. Train the model

```bash
python main.py
```

This will generate:
- mlruns/ (MLflow logs)
- final_lightgbm_pipeline.joblib (trained LightGBM pipeline)

### 4.7. Launch MLflow UI (optional)

```bash
mlflow ui
```

Open in browser:
http://127.0.0.1:5000

### 4.8. Run the Streamlit interface

```bash
python -m streamlit run app_streamlit.py
```

---

## 5. Baseline Model Summary

| Component      | Description                       |
|----------------|-----------------------------------|
| Features       | Original features                 |
| Preprocessing  | StandardScaler + OneHotEncoder    |
| Model          | RandomForest                      |
| F1             | ≈ 0.84                            |
| ROC-AUC        | ≈ 0.89                            |

This served as the baseline reference.

---

## 6. Experiment Tracking Summary

### We experimented with:
- Different model families
- Hyperparameters
- Feature engineering strategies
- Cleaned vs boosted datasets
- Ensemble stacking
 
### Impact:
- Engineered features improved F1
- Boosting models outperformed RandomForest
- LightGBM was the strongest model
- Stacking was not significantly superior
- Final performance:
    - **F1 = 0.8627**
    - **ROC-AUC = 0.9258**

---

## 7. Final Deliverables

- main.py – Final training pipeline
- final_lightgbm_pipeline.joblib – Saved LightGBM model pipeline
- ML_cleaned.ipynb – Clean experiment notebook
- eda.ipynb – Exploratory data analysis
- app.py – FastAPI app serving predictions
- app_streamlit.py – Streamlit app for persona testing
- mlruns/ – MLflow logs of all experiments
- Dataset files (raw, cleaned, engineered)

---

## 8. Business Use Case (BUC)

### [Insert Title]

[Fill in]