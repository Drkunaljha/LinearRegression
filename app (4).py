import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Titanic Survival", layout="centered")
st.title("🚢 Titanic Survival Prediction")

file = st.file_uploader("Upload Titanic CSV", type="csv")
if file:
    df = pd.read_csv(file)

    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df.dropna(subset=["Embarked"], inplace=True)

    df = pd.get_dummies(df, columns=["Sex","Embarked"], drop_first=True)

    features = ["Pclass","Age","SibSp","Parch","Fare","Sex_male","Embarked_Q","Embarked_S"]
    X = df[features]
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    st.success(f"Accuracy: {accuracy_score(y_test, pred):.2f}")
