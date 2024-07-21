import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Generate Synthetic Data
np.random.seed(42)

# Create three classes of points
banana = np.random.randn(50, 2) + [2, 2]
apple = np.random.randn(50, 2) + [7, 7]
watermelon = np.random.randn(50, 2) + [2, 7]

# Combine the data
data = np.vstack((banana, apple, watermelon))
labels = np.hstack((np.zeros(50), np.ones(50), np.full(50, 2)))

# Create a DataFrame
df = pd.DataFrame(data, columns=['Feature1', 'Feature2'])
df['Label'] = labels

# Step 2: Implement k-NN Algorithm
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(df, new_point, k=3):
    distances = []
    
    # Calculate distance from new_point to all other points
    for index, row in df.iterrows():
        dist = euclidean_distance(np.array([row['Feature1'], row['Feature2']]), new_point)
        distances.append((dist, row['Label']))
    
    # Sort distances and select the k nearest neighbors
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    
    # Determine the majority label
    labels = [label for _, label in k_nearest]
    prediction = max(set(labels), key=labels.count)
    
    return prediction

# Step 3: Visualize k-NN Predictions and Interactive Classification

# Generate a grid of points to classify
x_min, x_max = df['Feature1'].min() - 1, df['Feature1'].max() + 1
y_min, y_max = df['Feature2'].min() - 1, df['Feature2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

grid_points = np.c_[xx.ravel(), yy.ravel()]
predictions = np.array([knn_predict(df, point, k=3) for point in grid_points])
predictions = predictions.reshape(xx.shape)

# Plot the decision boundary
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(xx, yy, predictions, alpha=0.3, cmap='viridis')
scatter = ax.scatter(df['Feature1'], df['Feature2'], c=df['Label'], cmap='viridis', edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('k-NN Decision Boundary for Banana, Apple, and Watermelon (k=3)')

# Interactive classification
def onclick(event):
    ix, iy = event.xdata, event.ydata
    if ix is not None and iy is not None:
        new_point = np.array([ix, iy])
        prediction = knn_predict(df, new_point, k=3)
        label_name = ['Banana', 'Apple', 'Watermelon'][int(prediction)]
        print(f"Clicked point: ({ix:.2f}, {iy:.2f}) -> Predicted: {label_name}")
        ax.plot(ix, iy, 'ro')
        fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()