import pandas as pd
import joblib
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Create model folder
os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv("data/placement.csv")

print("Dataset loaded successfully")
print("Total rows:", len(df))

# Drop serial number if exists
if "sl_no" in df.columns:
    df.drop("sl_no", axis=1, inplace=True)

# Encode categorical columns
label_encoders = {}

for column in df.columns:
    if df[column].dtype == object:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# -------- FEATURE ENGINEERING --------

df["academic_avg"] = (
    df["ssc_p"] +
    df["hsc_p"] +
    df["degree_p"] +
    df["mba_p"]
) / 4

df["employability_score"] = (
    df["etest_p"] +
    df["mba_p"]
) / 2

# -------- FEATURE SELECTION --------

target = "status"

important_features = [
    "gender",
    "ssc_p",
    "hsc_p",
    "degree_p",
    "workex",
    "etest_p",
    "specialisation",
    "mba_p",
    "academic_avg",
    "employability_score"
]

X = df[important_features]
y = df[target]

# -------- GRID SEARCH --------

xgb = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

param_grid = {
    "n_estimators": [300, 500, 800],
    "learning_rate": [0.01, 0.03, 0.05],
    "max_depth": [3, 4, 5],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

# Train GridSearch
grid.fit(X, y)

# Best model
best_model = grid.best_estimator_

print("\nBest Accuracy:", grid.best_score_ * 100)
print("Best Parameters:", grid.best_params_)

# Save model and encoders
joblib.dump(best_model, "model/placement_model.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")

print("\nFinal optimized model saved successfully")
