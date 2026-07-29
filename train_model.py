import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE

# ----------------------------
# Load Dataset
# ----------------------------

df = pd.read_csv("data/predictive_maintenance.csv")

print("Dataset Loaded Successfully")
print(df.shape)

# ----------------------------
# Encode Machine Type
# ----------------------------

encoder = LabelEncoder()
df["Type"] = encoder.fit_transform(df["Type"])

# ----------------------------
# Features
# ----------------------------

X = df[[
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]]

# ----------------------------
# Target
# ----------------------------

y = df["Machine failure"]

# ----------------------------
# Train Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ----------------------------
# Balance Dataset
# ----------------------------

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)

print("\nBalanced Dataset:")
print(y_train.value_counts())

# ----------------------------
# Train Model
# ----------------------------

model = RandomForestClassifier(

    n_estimators=500,

    max_depth=15,

    min_samples_split=5,

    min_samples_leaf=2,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1

)

model.fit(X_train, y_train)

# ----------------------------
# Prediction
# ----------------------------

y_pred = model.predict(X_test)

# ----------------------------
# Accuracy
# ----------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy")
print(f"{accuracy:.2%}")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# ----------------------------
# Feature Importance
# ----------------------------

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print(importance)

# ----------------------------
# Save Model
# ----------------------------

joblib.dump(model, "models/model.pkl")
joblib.dump(encoder, "models/encoder.pkl")

print("\nModel Saved Successfully!")