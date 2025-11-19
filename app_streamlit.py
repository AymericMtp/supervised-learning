import streamlit as st
import pandas as pd
import joblib

# ==========================
# Global config
# ==========================

MODEL_PATH = "final_lightgbm_pipeline.joblib"
DATA_PATH = "income_boosted.csv"
TARGET_COL = "income"

# Explicit variable lists (updated)
CATEGORICAL_COLS = [
    "workclass",
    "education",
    "occupation",
    "race",
    "sex",
    "native-country",
]

NUMERIC_COLS = [
    "age",
    "education-num",
    "hours-per-week",
    "estimation_carriere_age",
    "is_married",
    "maried_education_experience",
]

# Mapping between education and education-num
EDU_TO_NUM = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16,
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_schema():
    df = pd.read_csv(DATA_PATH)
    return df.drop(columns=[TARGET_COL])


# Load model and schema
model = load_model()
X_schema = load_schema()

# Expected model features (correct order)
EXPECTED_FEATURES = X_schema.columns.tolist()

# Categorical & numeric UI lists
cat_cols = [c for c in CATEGORICAL_COLS if c in X_schema.columns]
numeric_cols = [c for c in NUMERIC_COLS if c in X_schema.columns]

# Numeric UI inputs (computed ones excluded)
numeric_user_input_cols = [
    col for col in numeric_cols
    if col not in ["education-num", "maried_education_experience"]
]

# ==========================
# Streamlit UI
# ==========================

st.title("🎯 Income Prediction (>50K / <=50K)")
st.write(
    "This app allows you to enter the demographic and professional information "
    "of a persona and estimate the probability that they earn more than 50K."
)

st.markdown("### 1️⃣ Enter the person's characteristics")

input_data = {}

# --------------------------
# Categorical variable inputs
# --------------------------
st.subheader("Categorical variables")

for col in cat_cols:
    options = sorted(X_schema[col].dropna().unique().tolist())
    if len(options) == 0:
        options = [""]
    input_data[col] = st.selectbox(f"{col}", options, index=0)

# --------------------------
# Numeric variable inputs
# --------------------------
st.subheader("Numeric variables")

for col in numeric_user_input_cols:
    if col == "is_married":
        married_bool = st.checkbox("Is married?", value=False)
        input_data[col] = 1 if married_bool else 0
    else:
        col_min = float(X_schema[col].min())
        col_max = float(X_schema[col].max())
        col_mean = float(X_schema[col].mean())
        input_data[col] = st.number_input(
            col,
            min_value=col_min,
            max_value=col_max,
            value=col_mean,
        )

# ==========================
# Derived features
# ==========================

st.markdown("### 2️⃣ Automatic derived features")

# Education-num from education
education_value = input_data.get("education")
education_num = EDU_TO_NUM.get(
    education_value,
    float(X_schema["education-num"].mean())
)
input_data["education-num"] = education_num

st.write(
    f"**Derived `education-num` from `education`:** "
    f"{education_value} → {education_num}"
)

# Derived interaction: maried_education_experience
age_val = float(input_data.get("age", 0))
is_married_val = float(input_data.get("is_married", 0))
maried_edu_exp = education_num * age_val * is_married_val

input_data["maried_education_experience"] = maried_edu_exp

st.write(
    f"**Derived `maried_education_experience`:** "
    f"{education_num} × {age_val} × {is_married_val} = {maried_edu_exp}"
)

# ==========================
# Preset capital-gain and capital-loss
# ==========================

input_data["capital-gain"] = 0
input_data["capital-loss"] = 0

st.info("Capital-gain and capital-loss are preset to 0 and hidden from the interface.")

# ==========================
# Prediction
# ==========================

st.markdown("### 3️⃣ Run prediction")

if st.button("Predict income"):
    row = {}

    for col in EXPECTED_FEATURES:
        if col in input_data:
            row[col] = input_data[col]
        else:
            # fallback: mean/mode from dataset
            if col in X_schema.columns:
                if X_schema[col].dtype == "object":
                    row[col] = X_schema[col].mode(dropna=True).iloc[0]
                else:
                    row[col] = float(X_schema[col].mean())

    df_input = pd.DataFrame([row])

    # Probability
    proba = model.predict_proba(df_input)[0, 1]
    pred_class = model.predict(df_input)[0]
    label = "> 50K" if pred_class == 1 else "<= 50K"
    proba_pct = proba * 100

    st.markdown("### 🧾 Prediction result")
    st.write(f"**Predicted income class:** `{label}`")
    st.write(f"**Probability of income > 50K:** `{proba_pct:.1f}%`")

    st.markdown("### 🔍 Features used for this prediction")
    st.dataframe(df_input)
