import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Perceptron:
    def __init__(self, input_size, lr, epochs):
        self.w = np.zeros(input_size)
        self.b = 0
        self.lr = lr
        self.epochs = epochs
        self.losses = []

    def fit(self, X_train, Y_train):
        for _ in range(self.epochs):
            for x_i in range(X_train.shape[0]):
                x = X_train[x_i]
                y = Y_train[x_i]
                y_pred = np.dot(x, self.w) + self.b
                error = y - y_pred

                self.w += (error * x * self.lr)
                self.b += (error * self.lr)

                loss = np.mean(np.abs(error))
                self.losses.append(loss)

    def predict(self, X_test):
        return np.dot(X_test, self.w) + self.b

    def plot_losses(self, X_train, Y_train, plot_3d=False, plot_3d_title='3D Plot'):
        if X_train.shape[1] == 1 and not plot_3d:
            plt.figure(figsize=(12, 6))
            plt.scatter(X_train, Y_train, color='blue', label='True Values')
            
            Y_pred = np.dot(X_train, self.w) + self.b
            plt.plot(X_train, Y_pred, color='red', label='Predicted Line')
            
            plt.title('2D Linear Regression Plot')
            plt.xlabel('Feature')
            plt.ylabel('Target')
            plt.legend()
            plt.show()
        
        elif X_train.shape[1] == 2 and plot_3d:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            X1 = X_train[:, 0]
            X2 = X_train[:, 1]
    
            ax.scatter(X1, X2, Y_train, color='blue', label='True Values')
            
            X1_grid, X2_grid = np.meshgrid(np.linspace(X1.min(), X1.max(), 100),
                                           np.linspace(X2.min(), X2.max(), 100))
            Z_pred = self.w[0] * X1_grid + self.w[1] * X2_grid + self.b
        
            ax.plot_surface(X1_grid, X2_grid, Z_pred, color='red', alpha=0.7)
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.set_zlabel("Target")
            ax.set_title(plot_3d_title)

        plt.figure(figsize=(10, 5))
        plt.plot(self.losses)
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Error')
        plt.grid(True)
        plt.show()



