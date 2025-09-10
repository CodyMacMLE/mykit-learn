## Base Formula
#### Mean Squared Error

$$J(w,b) = {1 \over2m}\sum_{n=1}^{m}(f_{w,b}(x^{(i)})-y^{(i)})^2 $$
<br>
#### Cross-Entropy Loss (Log Loss)

$$J(w,b) = -{1 \over m}(y^{T}log(\hat{y})+(1-y)^{T}log(1-\hat{y})) $$
<br>
#### Gradient Descent

$$ \theta_j=\theta_j - \alpha {\partial\over\partial\theta_j}J(\theta_0, \theta_1)$$
<br>

## Linear Regression

#### Formula of a line
$$ f(x)=wx+b $$
<br>
#### Cost of Linear Regression
$$ J(w,b) = {1 \over2m}\sum_{n=1}^{m}(w^Tx + b-y)^2  $$
<br>
#### Linear Gradient Descent
$$ w := w - \alpha{1 \over m}\sum_{n=1}^{m}(w^Tx + b-y)x  $$
$$ b := b - \alpha{1 \over m}\sum_{n=1}^{m}(w^Tx + b-y) $$
<br>

## Logistic Regression
#### Sigmoid Function
$$ \sigma(z)= {1 \over 1 + e^{-z}},\;\; z = w^Tx + b $$
<br>
$$ \hat{y}=\sigma(z) = {1 \over 1 + e^{-(w^Tx + b)}} $$
<br>
$$ {d \over dz}\hat{y} = {d \over dz} \sigma(z)= {\sigma(z)(1-\sigma(z))} $$
<br>

#### Cost of Logistic Regression
$$J(w,b) = -{1 \over m}(y^{T}log(\hat{y})+(1-y)^{T}log(1-\hat{y})) $$
<br>
<br>
$$ {\partial J \over \partial z} = {1\over m}(\hat{y}-y)$$
$$ {\partial J \over \partial w} = {1\over m}X^T(\hat{y}-y)$$
$$ {\partial J \over \partial b} = {1\over m}1^T(\hat{y}-y)$$

#### Logistic Gradient Descent
$$ w = w - \alpha{1\over m}X^T(\hat{y}-y)$$
$$ b = b - \alpha{1\over m}1^T(\hat{y}-y)$$

<br>
<br>
<br>