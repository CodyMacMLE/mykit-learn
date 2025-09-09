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
| ------------------------- |----------|
| Batch Gradient Descent    | 0.3208   |
| Mini-Batch GD (100)       | 7.8938   |
| Stochastic GD             | 10.6110  |
| Sklearn Linear Regression | 0.0115   |

---

## ⚙️ Custom Model Parameters

### Batch Gradient Descent

* α (Learning Rate): `0.001`
* Iterations: `50,000`
* Weights: `[7.3800, 17.6280, 0.3044, 0.8100, 0.5536]`
* Bias: `55.2111`

### Mini-Batch Gradient Descent (batch size = 100)

* α (Learning Rate): `0.001`
* Iterations: `10,000`
* Weights: `[7.3319, 17.5466, 0.3131, 0.8035, 0.5757]`
* Bias: `54.9438`

### Stochastic Gradient Descent

* α (Learning Rate): `0.001`
* Iterations: `50,000`
* Weights: `[7.3262, 17.6398, 0.3154, 0.8289, 0.5172]`
* Bias: `55.2709`

**Note:** All custom implementations break at cost difference < `epsilon = 0.000001` or max iterations.

---

## 🤖 Scikit-learn Model Parameters

**Linear Regression**

* Weights: `[7.3856, 17.6369, 0.3043, 0.8088, 0.5500]`
* Bias: `55.2408`

---

## 📉 Training Mean Squared Error (MSE)

| Model                     | MSE    |
| ------------------------- |--------|
| My Batch GD               | 4.0809 |
| My Mini-Batch GD          | 4.1569 |
| My Stochastic GD          | 4.0939 |
| Sklearn Linear Regression | 4.0826 |

---

## 🔮 Predictions (Sample)

| Sample | SkLearn ŷ | Batch GD ŷ | Mini-Batch GD ŷ | Stochastic GD ŷ |
| ------ |------------|-------------|------------------|------------------|
| 0      | 54.71      | 54.68       | 54.38            | 54.78            |
| 1      | 22.62      | 22.61       | 22.55            | 22.65            |
| 2      | 47.90      | 47.88       | 47.64            | 47.90            |
| 3      | 31.29      | 31.29       | 31.27            | 31.29            |
| 4      | 43.00      | 42.98       | 42.76            | 42.93            |

---

## 🚀 Conclusion

* All gradient descent approaches converged to **similar performance**.
* `Batch GD` yielded the lowest MSE with `Sci-Kit Linear Regression` second and the custom `Stochastic GD` third.
* Predictions are stable and consistent across methods.

---

## 📌 How to Run

1. Clone the repository

   ```bash
    git clone CodyMacMLE/mykit-learn
    conda env create -f environment.yml
    conda activate mykit-learn
   ```
