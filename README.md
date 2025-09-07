# 📊 Student Performance Prediction

This project explores **linear regression models** (custom implementations and scikit-learn) to predict a **Performance Index** based on study-related features.

---

## 📑 Dataset Summary

The dataset contains **10,000 records** with the following features:

| Feature                          | Mean   | Std    | Min | 25% | 50% | 75% | Max |
|----------------------------------|--------|--------|-----|-----|-----|-----|-----|
| Hours Studied                    | 4.99   | 2.59   | 1.0 | 3.0 | 5.0 | 7.0 | 9.0 |
| Previous Scores                  | 69.45  | 17.34  | 40  | 54  | 69  | 85  | 99  |
| Sleep Hours                      | 6.53   | 1.69   | 4.0 | 5.0 | 7.0 | 8.0 | 9.0 |
| Sample Question Papers Practiced | 4.58   | 2.87   | 0.0 | 2.0 | 5.0 | 7.0 | 9.0 |
| **Performance Index** (Target)   | 55.22  | 19.21  | 10  | 40  | 55  | 71  | 100 |

---

## ⚙️ Custom Model Parameters

### Batch Gradient Descent
- α (Learning Rate): `0.001`  
- Iterations: `20,000`  
- Weights: `0.0962`  
- Bias: `68.8305`  

### Mini-Batch Gradient Descent (batch size = 100)
- α (Learning Rate): `0.001`  
- Iterations: `20,000`  
- Weights: `0.0525`  
- Bias: `68.8009`  

### Stochastic Gradient Descent
- α (Learning Rate): `0.001`  
- Iterations: `20,000`  
- Weights: `0.0962`  
- Bias: `68.8305`  

---

## 🤖 Scikit-learn Model Parameters
**Linear Regression**  
- Weights: `-0.0726`  
- Bias: `69.8927`  

---

## 📉 Training Mean Squared Error (MSE)

| Model                       | MSE        |
|------------------------------|------------|
| My Batch GD                 | 300.82     |
| My Mini-Batch GD            | 300.66     |
| My Stochastic GD            | 300.82     |
| Sklearn Linear Regression   | 300.65     |

---

## 🔮 Predictions (Sample)

| Test Input (Hours Studied) | SkLearn ŷ | Batch GD ŷ | Mini-Batch GD ŷ | Stochastic GD ŷ |
|-----------------------------|------------|-------------|------------------|------------------|
| 5.0                         | 69.53      | 69.31       | 69.06            | 69.31            |
| 2.0                         | 69.75      | 69.02       | 68.91            | 69.02            |
| 7.0                         | 69.38      | 69.50       | 69.17            | 69.50            |
| 6.0                         | 69.46      | 69.40       | 69.12            | 69.40            |
| 7.0                         | 69.38      | 69.50       | 69.17            | 69.50            |

---

## ⚡ Complexity Analysis

The main training and prediction operations in the custom **`LinearRegression`** class scale linearly with respect to the number of samples and features:

- **Training Complexity**:  
  \[
  O(n * m * iterations)
  \]  

- **Prediction Complexity**:  
  \[
  O(n * m)
  \]  

where `n` = number of samples, `m` = number of features

This ensures the implementation remains efficient and scalable for large datasets.

---

## 🚀 Conclusion

- All gradient descent approaches converged to **similar performance**.  
- **Mini-Batch GD** and **Sklearn Linear Regression** yielded the lowest MSE.  
- Predictions are stable and consistent across methods.

---

## 📌 How to Run

1. Clone the repository  
   ```bash
    git clone CodyMacMLE/mykit-learn
    conda env create -f environment.yml
    conda activate mykit-learn
    ```