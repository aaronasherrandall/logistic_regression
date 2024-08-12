# Logistic Regression Example: Iris Dataset

This repository contains a simple example of applying logistic regression to a small dataset. The goal is to classify flowers as "Iris Setosa" or "Not Setosa" based on petal length and petal width.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Algebraic Method](#algebraic-method)
4. [Python Implementation](#python-implementation)
5. [Plotting the Data](#plotting-the-data)
6. [Results](#results)
7. [Conclusion](#conclusion)

## Introduction

Logistic Regression is a simple yet powerful classification algorithm commonly used for binary classification problems. This example demonstrates how logistic regression can be used to classify flowers in the Iris dataset.

## Dataset

The dataset used in this example consists of five data points with two features:

| Petal Length | Petal Width | Label         |
|--------------|-------------|---------------|
| 1            | 0           | 1 (Setosa)    |
| 4            | 1           | 0 (Not Setosa)|
| 2            | 0           | 1 (Setosa)    |
| 5            | 2           | 0 (Not Setosa)|
| 2            | 1           | 1 (Setosa)    |

## Algebraic Method

### Linear Combination

The linear combination in logistic regression is calculated as:

$$
z = w_1 \cdot x_1 + w_2 \cdot x_2 + b
$$

For example, with arbitrary weights $w_1 = 0.5$, $w_2 = -0.5$, and $b = 0$, for input $x_1 = 2$ and $x_2 = 1$:

$$
z = 0.5 \cdot 2 + (-0.5) \cdot 1 + 0 = 1 - 0.5 = 0.5
$$

### Sigmoid Function

The sigmoid function is applied to the linear combination $z$ to map it to a probability:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Applying it to $z = 0.5$:

$$
\sigma(0.5) = \frac{1}{1 + e^{-0.5}} \approx 0.6225
$$

### Cost Function (Log-Loss)

The cost function measures how well the model’s predictions match the actual labels.

For $y = 1$:

$$
\text{Cost} = -\log(0.6225) \approx 0.4748
$$

For $y = 0$:

$$
\text{Cost} = -\log(0.3775) \approx 0.9741
$$

### Gradient Descent

The model adjusts weights $w_1$, $w_2$, and bias $b$ iteratively to minimize the cost function:

$$
w_j := w_j - \alpha \frac{\partial J(\theta)}{\partial w_j}
$$

## Python Implementation

Here’s how to implement logistic regression using Python's scikit-learn:

```python
from sklearn.linear_model import LogisticRegression

# Training data
X_train = [[1, 0], [4, 1], [2, 0], [5, 2], [2, 1]]
y_train = [1, 0, 1, 0, 1]

# Initialize the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# The learned weights (coefficients) and bias (intercept)
print(f"Weights: {model.coef_}")
print(f"Bias: {model.intercept_}")
