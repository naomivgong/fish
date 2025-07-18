import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pickle


def create_model():
    # Download latest version
    fish_df = pd.read_csv("Fish.csv")

    #print(fish_df.isna().sum())
    fish_df = fish_df[fish_df["Species"].isin(["Bream", "Perch"])]
    x = fish_df.drop(columns = ["Species", "Category"])
    y = fish_df["Category"]

    fish_label = LabelEncoder()
    y = fish_label.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
    lr = LogisticRegression()

    lr.fit(x_train, y_train)
    # y_preds_lr = lr.predict(x_test)
    # print(y_preds_lr)
    # lr_accuracy = accuracy_score(y_test, y_preds_lr)
    # print("The accuracy score of Logistic Regresion is ", lr_accuracy)
    return lr, fish_label


def predict_species(model, label_encoder, input_data):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict_proba(input_df)
    df_prediction_probs = pd.DataFrame(prediction)
    df_prediction_probs.columns = ["Bream_Probability", "Perch_Probability"]
    df_prediction_probs.rename(columns={0: "Bream_Probability", 1: "Perch_Probability"})
    return df_prediction_probs

if __name__ == "__main__":
    filename = 'fish_model.sav'
    model, label_encoder = create_model()
    pickle.dump(model, open(filename, 'wb'))
