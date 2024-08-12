import matplotlib.pyplot as plt

# The dataset
X_train = [[1,0], [4,1], [2,0], [5,2], [2,1]]
y_train = [1, 0, 1, 0, 1]

# Convert data to separate lists for plotting
x1 = [point[0] for point in X_train] # Petal Lengths
x2 = [point[1] for point in X_train] # Petal Widths

# Create the plot
plt.figure(figsize=(8,6))

# Plot each point and color based on the label
for i in range(len(X_train)):
    # if the flower is labeled as Setosa
    if y_train[i] == 1:
        plt.scatter(x1[i], x2[i], color="blue", label='Setosa' if i == 0 else "")
    else:
        plt.scatter(x1[i], x2[i], color="red", label='Not Setosa' if i == 0 else "")

# Labeling the plot
plt.title('Petel Length vs Petel Width')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()