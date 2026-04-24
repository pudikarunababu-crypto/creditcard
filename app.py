import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

st.title("💳 Credit Card Fraud Detection App")

# Load dataset
data = pd.read_csv("cc_info.csv")   # <-- change if needed

# Show data
st.subheader("Dataset Preview")
st.dataframe(data.head())

# Show columns
st.subheader("Columns in Dataset")
st.write(data.columns)

# 🔴 FIX: Check correct column name
target_col = None

for col in data.columns:
    if col.lower() in ['class', 'target', 'label']:
        target_col = col

if target_col is None:
    st.error("❌ No target column (Class/target/label) found in dataset")
    st.stop()

st.success(f"Using target column: {target_col}")

# Value counts
st.subheader("Class Distribution")
st.write(data[target_col].value_counts())

fig1, ax1 = plt.subplots()
data[target_col].value_counts().plot(kind='bar', ax=ax1)
ax1.set_title("Before SMOTE")
st.pyplot(fig1)

# Split data
X = data.drop(target_col, axis=1)
Y = data[target_col]

# SMOTE
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, Y)

st.subheader("After SMOTE")
fig2, ax2 = plt.subplots()
pd.Series(y_res).value_counts().plot(kind='bar', ax=ax2)
ax2.set_title("After SMOTE")
st.pyplot(fig2)

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Results
st.subheader("Model Performance")
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

# Prediction demo
st.subheader("Sample Predictions")

normal = X.iloc[0]
fraud = X.iloc[-1]

st.write("Normal Prediction:", model.predict([normal]))
st.write("Fraud Prediction:", model.predict([fraud]))
