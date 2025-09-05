# External Libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Internal Libraries
import mykitlearn

# Panda Settings
pd.set_option('display.max_columns', None)

def main():
    # Dataset: Experience to Salary
    df = pd.read_csv('Salary_Data.csv')

    print("---- Data Summary ----")
    print(df.describe())


    # 1D array for my custom GD Model
    X = df.iloc[:, 0].to_numpy(dtype=float).reshape(-1, 1)
    y = df.iloc[:, 1].to_numpy(dtype=float)

    test_X = [0.25, 1.6, 24.3, 44.2, 31.3, 4.2, 3.8, 2.6, 7.7, 7.2, 6.9]

    # ---- 1. Train my custom gradient descent model ----
    my_batch_model = mykitlearn.LinearRegression(learning_rate=0.001, iterations=20000, gd_type="BGd", batches=1)
    my_batch_model.train(X, y)
    y_pred_myBatch = my_batch_model.predict(X)

    my_miniBatch_model = mykitlearn.LinearRegression(learning_rate=0.001, iterations=20000, gd_type="BGd", batches=2)
    my_miniBatch_model.train(X, y)
    y_pred_miniBatch = my_miniBatch_model.predict(X)

    my_stochastic_model = mykitlearn.LinearRegression(learning_rate=0.001, iterations=20000, gd_type="SGd", batches=1)
    my_stochastic_model.train(X, y)
    y_pred_stochastic = my_stochastic_model.predict(X)


    print("\n----- Custom Model Parameters -----")
    print("Batch GD:              ", my_batch_model)
    print("Mini-batch GD (2):     ", my_miniBatch_model)
    print("Stochastic GD:         ", my_stochastic_model)

    # ---- 2. Train with scikit-learn ----
    model_sk = LinearRegression()
    model_sk.fit(X, y)
    y_pred_sk = model_sk.predict(X)

    print("\n----- Scikit-learn Model Parameters -----")
    print("Linear Regression:                            w:", model_sk.coef_, " b:", model_sk.intercept_)

    # ---- 3. Plot results ----

    plt.scatter(X, y, color="blue", label="Data")
    plt.plot(X, y_pred_myBatch, color="red", label="Mykit-learn")
    plt.plot(X, y_pred_miniBatch, color="blue",label="Scikit-learn")
    plt.plot(X, y_pred_stochastic, color="orange",label="Scikit-learn")
    plt.plot(X, y_pred_sk, color="green", linestyle="--", label="Scikit-learn")
    plt.legend()
    plt.title("Linear Regression Comparison")
    plt.show()

    # ---- 4. Prediction Results ----
    tx = np.asarray(test_X, dtype=float).ravel()

    y_pred_gd_flat = np.asarray(my_batch_model.predict(tx)).ravel()
    y_pred_miniBatch_flat = np.asarray(my_miniBatch_model.predict(tx)).ravel()
    y_pred_stochastic_flat = np.asarray(my_stochastic_model.predict(tx)).ravel()
    y_pred_sk_flat = model_sk.predict(tx.reshape(-1, 1)).ravel()

    # 3) Build the table (now every column is 1-D with the same length)
    results = pd.DataFrame({
        "Test Inputs": tx,
        "SkLearn y_pred": y_pred_sk_flat,
        "Batch y_pred": y_pred_gd_flat,
        "MiniBatch y_pred": y_pred_miniBatch_flat,
        "Stochastic y_pred": y_pred_stochastic_flat,
    })

    print("\n",results)


if __name__ == "__main__":
    main()