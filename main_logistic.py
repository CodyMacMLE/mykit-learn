# External Libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

import mykitlearn
# Internal Libraries
from mykitlearn import LogisticRegression

# Panda Settings
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

def main():
    # Import Dataset "Predicting_HeartDisease.csv"
    # https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression
    df = pd.read_csv('data/Predicting_HeartDisease.csv')
    df_cleaned = df.dropna()

    # Summarize data
    print("---- Data Summary ----")
    print(df_cleaned.describe())

    print("\n---- Data Head ----")
    print(df_cleaned.head())

    # Split data sets into respective X / y and normalize
    X = df_cleaned.iloc[:, :-1]
    X_norm = preprocessing.scale(X)
    y = df_cleaned.iloc[:, -1]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2)

    # Training models
    myBatchModel = mykitlearn.LogisticRegression()
    myMiniBatchModel = mykitlearn.LogisticRegression(batches=4)
    myStochasticModel = mykitlearn.LogisticRegression(gd_type="SGd")
    sklearnModel = LogisticRegression()

    print("\n----- Training Time -----")

    # Batch Model
    batchStartTime = time.time()
    myBatchModel.train(X_train, y_train)
    batchEndTime = time.time()
    batchTime = batchEndTime - batchStartTime
    print(f"My Batch Model Training Time:      {batchTime:.4f}s")

    # Mini-Batch Model
    miniBatchStartTime = time.time()
    myMiniBatchModel.train(X_train, y_train)
    miniBatchEndTime = time.time()
    miniBatchTime = miniBatchEndTime - miniBatchStartTime
    print(f"My Mini-Batch Model Training Time: {miniBatchTime:.4f}s")

    # Stochastic Model
    stochasticStartTime = time.time()
    myStochasticModel.train(X_train, y_train)
    stochasticEndTime = time.time()
    stochasticTime = stochasticEndTime - stochasticStartTime
    print(f"My Stochastic Model Training Time: {stochasticTime:.4f}s")

    # Sklearn Model
    sklearnStartTime = time.time()
    sklearnModel.train(X_train, y_train)
    sklearnEndTime = time.time()
    sklearnTime = sklearnEndTime - sklearnStartTime
    print(f"Sklearn Model Training Time:       {sklearnTime:.4f}s")


    # Testing Errors
    y_pred_batch = myBatchModel.predict(X_test, proba=False)
    y_pred_minibatch = myMiniBatchModel.predict(X_test, proba=False)
    y_pred_stochastic = myStochasticModel.predict(X_test, proba=False)
    y_pred_sklearn = sklearnModel.predict(X_test).astype(np.int64)
    error_batch = 1 - accuracy_score(y_test, y_pred_batch)
    error_minibatch = 1 - accuracy_score(y_test, y_pred_minibatch)
    error_stochastic = 1 - accuracy_score(y_test, y_pred_stochastic)
    error_sklearn = 1 - accuracy_score(y_test, y_pred_sklearn)

    print("\n----- Errors -----")
    print("My Batch:     ", error_batch)
    print("My Mini-Batch:", error_minibatch)
    print("My Stochastic:", error_stochastic)
    print("Sklearn:      ", error_sklearn)


if __name__ == "__main__":
    main()