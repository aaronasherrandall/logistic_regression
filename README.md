Logistic Regression Example: Iris Dataset
This repository contains a simple example of applying logistic regression to a small dataset. The goal is to classify flowers as "Iris Setosa" or "Not Setosa" based on petal length and petal width.

Table of Contents
Introduction
Dataset
Algebraic Method
Python Implementation
Plotting the Data
Results
Conclusion
Introduction
Logistic Regression is a simple yet powerful classification algorithm commonly used for binary classification problems. This example demonstrates how logistic regression can be used to classify flowers in the Iris dataset.

Dataset
The dataset used in this example consists of five data points with two features:

Petal Length	Petal Width	Label
1	0	1 (Setosa)
4	1	0 (Not Setosa)
2	0	1 (Setosa)
5	2	0 (Not Setosa)
2	1	1 (Setosa)
Algebraic Method
Linear Combination
The linear combination in logistic regression is calculated as:

𝑧
=
𝑤
1
⋅
𝑥
1
+
𝑤
2
⋅
𝑥
2
+
𝑏
z=w 
1
​
 ⋅x 
1
​
 +w 
2
​
 ⋅x 
2
​
 +b
For example, with arbitrary weights 
𝑤
1
=
0.5
w 
1
​
 =0.5, 
𝑤
2
=
−
0.5
w 
2
​
 =−0.5, and 
𝑏
=
0
b=0, for input 
𝑥
1
=
2
x 
1
​
 =2 and 
𝑥
2
=
1
x 
2
​
 =1:

𝑧
=
0.5
⋅
2
+
(
−
0.5
)
⋅
1
+
0
=
1
−
0.5
=
0.5
z=0.5⋅2+(−0.5)⋅1+0=1−0.5=0.5
Sigmoid Function
The sigmoid function is applied to the linear combination 
𝑧
z to map it to a probability:

𝜎
(
𝑧
)
=
1
1
+
𝑒
−
𝑧
σ(z)= 
1+e 
−z
 
1
​
 
Applying it to 
𝑧
=
0.5
z=0.5:

𝜎
(
0.5
)
=
1
1
+
𝑒
−
0.5
≈
0.6225
σ(0.5)= 
1+e 
−0.5
 
1
​
 ≈0.6225
Cost Function (Log-Loss)
The cost function measures how well the model’s predictions match the actual labels. For 
𝑦
=
1
y=1:

Cost
=
−
log
⁡
(
0.6225
)
≈
0.4748
Cost=−log(0.6225)≈0.4748
For 
𝑦
=
0
y=0:

Cost
=
−
log
⁡
(
0.3775
)
≈
0.9741
Cost=−log(0.3775)≈0.9741
Gradient Descent
The model adjusts weights 
𝑤
1
w 
1
​
 , 
𝑤
2
w 
2
​
 , and bias 
𝑏
b iteratively to minimize the cost function:

𝑤
𝑗
:
=
𝑤
𝑗
−
𝛼
∂
𝐽
(
𝜃
)
∂
𝑤
𝑗
w 
j
​
 :=w 
j
​
 −α 
∂w 
j
​
 
∂J(θ)
​
 
Python Implementation
Here’s how to implement logistic regression using Python's scikit-learn:

python
Copy code
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
Output:
plaintext
Copy code
Weights: [[-0.79259092 -0.30844669]]
Bias: [1.93381652]
Interpretation:
Weights: These values correspond to the features in your training data:
𝑤
1
=
−
0.79259092
w 
1
​
 =−0.79259092 (Petal Length)
𝑤
2
=
−
0.30844669
w 
2
​
 =−0.30844669 (Petal Width)
Bias: The bias shifts the decision boundary, making the model more likely to predict class 1 (Setosa) when the input features are neutral.
Accuracy
After training the model, you can evaluate its accuracy:

python
Copy code
# Test data
X_test = [[1, 0], [4, 1], [2, 0], [5, 2], [2, 1]]
y_test = [1, 0, 1, 0, 1]

# Predict the labels for the test set
y_pred = model.predict(X_test)

# Calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
Output:
plaintext
Copy code
Accuracy: 1.0
This indicates that the model made perfect predictions on the test data.

Plotting the Data
To visualize the data on a Cartesian plane:

python
Copy code
import matplotlib.pyplot as plt

# Your dataset
X_train = [[1, 0], [4, 1], [2, 0], [5, 2], [2, 1]]
y_train = [1, 0, 1, 0, 1]

# Convert data to separate lists for plotting
x1 = [point[0] for point in X_train]  # Petal Lengths
x2 = [point[1] for point in X_train]  # Petal Widths

# Create the plot
plt.figure(figsize=(8, 6))

# Plot each point and color based on the label
for i in range(len(X_train)):
    if y_train[i] == 1:
        plt.scatter(x1[i], x2[i], color='blue', label='Setosa' if i == 0 else "")
    else:
        plt.scatter(x1[i], x2[i], color='red', label='Not Setosa' if i == 0 else "")

# Labeling the plot
plt.title('Petal Length vs Petal Width')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
This will generate a scatter plot where points labeled as 1 are colored blue, and points labeled as 0 are colored red.

Results
The model achieved an accuracy of 1.0, meaning it correctly classified all the test data points. The scatter plot visually demonstrates the data distribution and how well the logistic regression model fits the data.
