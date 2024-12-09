import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class Perceptron:
    def __init__(self, learning_rate, input_length):
        self.learning_rate = learning_rate
        self.weights = np.random.uniform(-0.1, 0.1, input_length)
        self.bias = np.random.uniform(-0.1, 0.1)

    def activation(self, x, function='linear'):
        if function == 'linear':
            return x
        else:
            raise Exception("Only 'linear' activation is supported for regression.")

    def fit(self, X_train, y_train, X_test, y_test, epochs, function='linear'):
        train_losses, test_losses = [], []

        for epoch in tqdm(range(epochs)):
            for x, y in zip(X_train, y_train):
                y_pred = x @ self.weights + self.bias
                y_pred = self.activation(y_pred, function)

                error = y - y_pred
                self.weights += self.learning_rate * error * x
                self.bias += self.learning_rate * error

            train_loss = self.calc_loss(X_train, y_train)
            test_loss = self.calc_loss(X_test, y_test)

            train_losses.append(train_loss)
            test_losses.append(test_loss)

        self.plot_losses(train_losses, test_losses)
        
    def predict(self, X):
        y_pred = []
        for x in X:
            y = x @ self.weights + self.bias
            y_pred.append(self.activation(y))
        return np.array(y_pred)

    def calc_loss(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)

    def plot_losses(self, train_losses, test_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.title('Loss over Epochs (Weather)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.show()
    
    def evaluate(self, X_test, y_test, function='linear'):
        y_pred = self.predict(X_test)
        mse = np.mean((y_test- y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))

        return {
            "MSE": mse,
            "MAE" : mae
        }

    def predict2(self, X, day_num):
        if day_num < 0 or day_num >= len(X):
            raise ValueError("Day number is out of range. Provide a valid day index.")
        
        day_feature = X[day_num]
        y_pred = day_feature @ self.weights + self.bias
        return y_pred


