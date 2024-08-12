from sklearn.linear_model import LogisticRegression

# training data
X_train = [[1,0], [4,1], [5,2]]
y_train = [1, 0, 0]

# Initialize the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# The learned weights (coefficients) and bias (intercept)
print(f"Weights: {model.coef_}")
print(f"Bias: {model.intercept_}")

# Testing data
X_test = [[2,0], [2,1]]
y_test = [1,1]

# Predict the labels for the test set
y_pred = model.predict(X_test)

# Predict the labels for the test set
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")