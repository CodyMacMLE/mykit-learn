# External Libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Internal Libraries
import mykitlearn

# Panda Settings
pd.set_option('display.max_columns', None)

def main():
    # Dataset: Experience to Salary
    df = pd.read_csv('Student_Performance.csv')

    print("---- Data Summary ----")
    print(df.describe())


    # 1D array for my custom GD Model
    X = df.iloc[:, 0].to_numpy(dtype=float).reshape(-1, 1)
    y = df.iloc[:, 1].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---- 1. Train my custom gradient descent model ----
    my_batch_model = mykitlearn.LinearRegression(learning_rate=0.001, iterations=20000, gd_type="BGd", batches=1)
    my_batch_model.train(X_train, y_train)
    y_pred_myBatch = my_batch_model.predict(X_train)

    my_miniBatch_model = mykitlearn.LinearRegression(learning_rate=0.001, iterations=20000, gd_type="BGd", batches=100)
    my_miniBatch_model.train(X_train, y_train)
    y_pred_miniBatch = my_miniBatch_model.predict(X_train)

    my_stochastic_model = mykitlearn.LinearRegression(learning_rate=0.001, iterations=20000, gd_type="SGd", batches=1)
    my_stochastic_model.train(X_train, y_train)
    y_pred_stochastic = my_stochastic_model.predict(X_train)


    print("\n----- Custom Model Parameters -----")
    print("Batch GD:                ", my_batch_model)
    print("Mini-batch GD (100):     ", my_miniBatch_model)
    print("Stochastic GD:           ", my_stochastic_model)

    # ---- 2. Train with scikit-learn ----
    model_sk = LinearRegression()
    model_sk.fit(X_train, y_train)
    y_pred_sk = model_sk.predict(X_train)

    print("\n----- Scikit-learn Model Parameters -----")
    print("Linear Regression:                              w:", model_sk.coef_, " b:", model_sk.intercept_)

    # ---- 3. Plot results ----

    plt.scatter(X_train, y_train, color="blue", label="Data")
    plt.plot(X_train, y_pred_myBatch, color="red", label="Mykit-learn")
    plt.plot(X_train, y_pred_miniBatch, color="blue",label="Scikit-learn")
    plt.plot(X_train, y_pred_stochastic, color="orange",label="Scikit-learn")
    plt.plot(X_train, y_pred_sk, color="green", linestyle="--", label="Scikit-learn")
    plt.legend()
    plt.title("Linear Regression Comparison")
    plt.show()

    print("\n---- Training MSE ----")
    # Calculate MSE for each model
    mse_my_batch = mean_squared_error(y_test, my_batch_model.predict(X_test))
    mse_my_miniBatch = mean_squared_error(y_test, my_miniBatch_model.predict(X_test))
    mse_my_stochastic = mean_squared_error(y_test, my_stochastic_model.predict(X_test))
    mse_sk = mean_squared_error(y_test, model_sk.predict(X_test))

    print("MSE - My Batch GD:", mse_my_batch)
    print("MSE - My MiniBatch GD:", mse_my_miniBatch)
    print("MSE - My Stochastic GD:", mse_my_stochastic)
    print("MSE - Sklearn Linear Regression:", mse_sk)

    # ---- 4. Prediction Results ----
    tx = np.asarray(X_test, dtype=float).ravel()

    y_pred_gd_flat = np.asarray(my_batch_model.predict(tx)).ravel()
    y_pred_miniBatch_flat = np.asarray(my_miniBatch_model.predict(tx)).ravel()
    y_pred_stochastic_flat = np.asarray(my_stochastic_model.predict(tx)).ravel()
    y_pred_sk_flat = model_sk.predict(tx.reshape(-1, 1)).ravel()

    # 3) Build the table (now every column is 1-D with the same length)
    results = pd.DataFrame({
        "Test Inputs": tx[:5],
        "SkLearn y_pred": y_pred_sk_flat[:5],
        "Batch y_pred": y_pred_gd_flat[:5],
        "MiniBatch y_pred": y_pred_miniBatch_flat[:5],
        "Stochastic y_pred": y_pred_stochastic_flat[:5],
    })

    print("\n---- Predict ----\n",results)



if __name__ == "__main__":
    main()