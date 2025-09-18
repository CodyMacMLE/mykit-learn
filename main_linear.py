# External Libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time

# Internal Libraries
from mykitlearn import LinearRegression

# Panda Settings
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

def main():
    # Import Dataset "Student_Performance.csv"
    df = pd.read_csv('data/Student_Performance.csv')
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map({'Yes': 1, 'No': 0})

    print("---- Data Summary ----")
    print(df.describe())

    print("\n---- Data Head ----")
    print(df.head())


    # 1D array for my custom GD Model
    X = df.iloc[:, 0:5].to_numpy(dtype=float)
    X_norm = (X - X.mean(axis=0)) / X.std(axis=0)  # Normalize features

    y = df["Performance Index"].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)


    # ---- 1. Train my custom gradient descent model ----
    batch_start = time.time()
    my_batch_model = mykitlearn.LinearRegression(learning_rate=0.001, iterations=50000, gd_type="BGd", batches=1)
    my_batch_model.train(X_train, y_train)
    batch_end = time.time()

    miniBatch_start = time.time()
    my_miniBatch_model = mykitlearn.LinearRegression(learning_rate=0.001, iterations=10000, gd_type="BGd", batches=100)
    my_miniBatch_model.train(X_train, y_train)
    miniBatch_end = time.time()

    stochastic_start = time.time()
    my_stochastic_model = mykitlearn.LinearRegression(learning_rate=0.001, iterations=50000, gd_type="SGd", batches=1)
    my_stochastic_model.train(X_train, y_train)
    stochastic_end = time.time()

    # ---- 2. Train with scikit-learn ----

    sklearn_start = time.time()
    model_sk = LinearRegression()
    model_sk.fit(X_train, y_train)
    sklearn_end = time.time()

    # ---- 3. Model Parameters & MSE ----

    print("\n----- Training Time -----")
    print(f"Batch GD:                  {batch_end - batch_start:.4f}s")
    print(f"Mini-batch GD (100):       {miniBatch_end - miniBatch_start:.4f}s")
    print(f"Stochastic GD:             {stochastic_end - stochastic_start:.4f}s")
    print(f"SKLearn Linear Regression: {sklearn_end - sklearn_start:.4f}s")

    print("\n----- Custom Model Parameters -----")
    print("Batch GD:                ", my_batch_model)
    print("Mini-batch GD (100):     ", my_miniBatch_model)
    print("Stochastic GD:           ", my_stochastic_model)
    print("SKLearn Linear Regression:                    w:", model_sk.coef_, " b:", model_sk.intercept_)

    print("\n---- Training MSE ----")
    # Calculate MSE for each model
    mse_my_batch = mean_squared_error(y_test, my_batch_model.predict(X_test))
    mse_my_miniBatch = mean_squared_error(y_test, my_miniBatch_model.predict(X_test))
    mse_my_stochastic = mean_squared_error(y_test, my_stochastic_model.predict(X_test))
    mse_sk = mean_squared_error(y_test, model_sk.predict(X_test))

    print("MSE - My Batch GD:              ", mse_my_batch)
    print("MSE - My MiniBatch GD:          ", mse_my_miniBatch)
    print("MSE - My Stochastic GD:         ", mse_my_stochastic)
    print("MSE - Sklearn Linear Regression:", mse_sk)

    # ---- 4. Prediction Results ----
    y_pred_gd_flat = np.asarray(my_batch_model.predict(X_test)).ravel()
    y_pred_miniBatch_flat = np.asarray(my_miniBatch_model.predict(X_test)).ravel()
    y_pred_stochastic_flat = np.asarray(my_stochastic_model.predict(X_test)).ravel()
    y_pred_sk_flat = model_sk.predict(X_test).ravel()

    # 3) Build the table (now every column is 1-D with the same length)
    results = pd.DataFrame({
        "SkLearn y_pred": y_pred_sk_flat[:5],
        "Batch y_pred": y_pred_gd_flat[:5],
        "MiniBatch y_pred": y_pred_miniBatch_flat[:5],
        "Stochastic y_pred": y_pred_stochastic_flat[:5],
    })

    print("\n---- Predict ----\n",results)



if __name__ == "__main__":
    main()