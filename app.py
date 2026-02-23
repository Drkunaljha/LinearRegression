import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic Survival Prediction App")
st.write("Upload Titanic dataset and check survival prediction.")

file = st.file_uploader("Upload Titanic CSV", type="csv")

if file:
    df = pd.read_csv(file)

    # Data Cleaning
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df.dropna(subset=["Embarked"], inplace=True)

    # Encoding
    df = pd.get_dummies(df, columns=["Sex","Embarked"], drop_first=True)

    features = ["Pclass","Age","SibSp","Parch","Fare","Sex_male","Embarked_Q","Embarked_S"]
    X = df[features]
    y = df["Survived"]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, pred)
    st.success(f"Model Accuracy: {acc:.2f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, pred))

    # Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, pred))

    # -------------------------
    # 🔹 Manual Prediction Section
    # -------------------------
    st.subheader("Check Survival for New Passenger")

    pclass = st.selectbox("Passenger Class", [1,2,3])
    age = st.slider("Age", 1, 80, 25)
    sibsp = st.number_input("Siblings/Spouses", 0, 10, 0)
    parch = st.number_input("Parents/Children", 0, 10, 0)
    fare = st.number_input("Fare", 0.0, 500.0, 50.0)
    sex = st.selectbox("Sex", ["Male","Female"])
    embarked = st.selectbox("Embarked", ["Q","S"])

    if st.button("Predict"):
        sex_male = 1 if sex == "Male" else 0
        embarked_Q = 1 if embarked == "Q" else 0
        embarked_S = 1 if embarked == "S" else 0

        input_data = np.array([[pclass, age, sibsp, parch, fare,
                                sex_male, embarked_Q, embarked_S]])

        input_data = scaler.transform(input_data)
        result = model.predict(input_data)

        if result[0] == 1:
            st.success("🎉 Passenger Survived!")
        else:
            st.error("❌ Passenger Did Not Survive.")
