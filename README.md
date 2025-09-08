# 📊 Student Performance Prediction

This project explores **linear regression models** (custom implementations and scikit-learn) to predict a **Performance Index** based on study-related features.

---

## 📑 Dataset Summary

The dataset contains **10,000 records** with the following features:

| Feature                          | Mean  | Std   | Min | 25% | 50% | 75% | Max |
| -------------------------------- | ----- | ----- | --- | --- | --- | --- | --- |
| Hours Studied                    | 4.99  | 2.59  | 1.0 | 3.0 | 5.0 | 7.0 | 9.0 |
| Previous Scores                  | 69.45 | 17.34 | 40  | 54  | 69  | 85  | 99  |
| Extracurricular Activities       | 0.49  | 0.50  | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| Sleep Hours                      | 6.53  | 1.69  | 4.0 | 5.0 | 7.0 | 8.0 | 9.0 |
| Sample Question Papers Practiced | 4.58  | 2.87  | 0.0 | 2.0 | 5.0 | 7.0 | 9.0 |
| **Performance Index** (Target)   | 55.22 | 19.21 | 10  | 40  | 55  | 71  | 100 |

---

## ⚙️ Training Time

| Model                     | Time (s) |
| ------------------------- | -------- |
| Batch Gradient Descent    | 1.7268   |
| Mini-Batch GD (100)       | 10.3956  |
| Stochastic GD             | 1.7038   |
| Sklearn Linear Regression | 0.0107   |

---

## ⚙️ Custom Model Parameters

### Batch Gradient Descent

* α (Learning Rate): `0.001`
* Iterations: `50,000`
* Weights: `[7.3856, 17.6369, 0.3043, 0.8088, 0.5500]`
* Bias: `55.2408`

### Mini-Batch Gradient Descent (batch size = 100)

* α (Learning Rate): `0.001`
* Iterations: `10,000`
* Weights: `[7.3770, 17.6385, 0.3027, 0.8001, 0.5505]`
* Bias: `55.2390`

### Stochastic Gradient Descent

* α (Learning Rate): `0.001`
* Iterations: `50,000`
* Weights: `[7.3856, 17.6369, 0.3043, 0.8088, 0.5500]`
* Bias: `55.2408`

---

## 🤖 Scikit-learn Model Parameters

**Linear Regression**

* Weights: `[7.3856, 17.6369, 0.3043, 0.8088, 0.5500]`
* Bias: `55.2408`

---

## 📉 Training Mean Squared Error (MSE)

| Model                     | MSE  |
| ------------------------- | ---- |
| My Batch GD               | 4.08 |
| My Mini-Batch GD          | 4.08 |
| My Stochastic GD          | 4.08 |
| Sklearn Linear Regression | 4.08 |

---

## 🔮 Predictions (Sample)

| Sample | SkLearn ŷ | Batch GD ŷ | Mini-Batch GD ŷ | Stochastic GD ŷ |
| ------ | ---------- | ----------- | ---------------- | ---------------- |
| 0      | 54.71      | 54.71       | 54.70            | 54.71            |
| 1      | 22.62      | 22.62       | 22.63            | 22.62            |
| 2      | 47.90      | 47.90       | 47.89            | 47.90            |
| 3      | 31.29      | 31.29       | 31.27            | 31.29            |
| 4      | 43.00      | 43.00       | 43.01            | 43.00            |

---

## ⚡ Complexity Analysis

The main training and prediction operations in the custom **`LinearRegression`** class scale linearly with respect to the number of samples and features:

* **Training Complexity**:

  $$
  O(n * m * i)
  $$

* **Prediction Complexity**:

  $$
  O(n * m)
  $$

where `n` = number of samples, `m` = number of features, and `i` = number of iterations.

This ensures the implementation remains efficient and scalable for large datasets.

---

## 🚀 Conclusion

* All gradient descent approaches converged to **similar performance**.
* **Batch GD** and **Stochastic GD** yielded the lowest MSE.
* Predictions are stable and consistent across methods.

---

## 📌 How to Run

1. Clone the repository

   ```bash
    git clone CodyMacMLE/mykit-learn
    conda env create -f environment.yml
    conda activate mykit-learn
   ```
