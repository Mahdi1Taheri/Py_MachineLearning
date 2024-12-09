import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
import seaborn as sns

class Perceptron:
    def __init__(self, learning_rate, input_length):
        self.learning_rate = learning_rate
        self.weights = np.random.uniform(-0.1, 0.1, input_length)
        self.bias = np.random.uniform(-0.1, 0.1)

    def confusion_matrix_values(self,X_test, y_test):
        y_pred = self.predict(X_test)
        y_pred = np.where(y_pred > .5, 1, 0)
        cm = confusion_matrix(y_test,y_pred)
        cm_display = ConfusionMatrixDisplay(cm,display_labels=[0,1])

        cm_display.plot()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    def activation(self, x, function):
        '''function parameters:
        * `sigmoid`
        * `relu`
        * `tanh`
        * `linear`
         '''
        if function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif function == 'relu':
            return np.maximum(0, x)
        elif function == 'tanh':
            return np.tanh(x)
        elif function == 'linear':
            return x
        else:
            raise Exception('Unknown Activation function')

    def fit(self, X_train, Y_train, X_test, Y_test, epochs, function):
        '''function parameters:
        * `sigmoid`
        * `relu`
        * `tanh`
        * `linear`
         '''

        train_losses, test_losses = [], []
        train_accuracies, test_accuracies = [], []

        for epoch in tqdm(range(epochs)):
            for x, y in zip(X_train, Y_train):
                y_pred = x @ self.weights + self.bias
                y_pred = self.activation(y_pred, function)
                error = y - y_pred

                self.weights += self.learning_rate * error * x
                self.bias += self.learning_rate * error

            train_loss = self.calc_loss(X_train, Y_train, 'mse')
            test_loss = self.calc_loss(X_test, Y_test, 'mse')
            train_accuracy = self.calc_accuracy(X_train, Y_train)
            test_accuracy = self.calc_accuracy(X_test, Y_test)

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

        self.plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies)

    def predict(self, X_test):
        Y_pred = []
        for x_test in X_test:
            y_pred = x_test @ self.weights + self.bias
            y_pred = self.activation(y_pred, 'linear') 
            y_pred = (y_pred >= 0.5).astype(int)  
            Y_pred.append(y_pred)
        return np.array(Y_pred).flatten()

    def calc_loss(self, X, Y, metric):
        '''choose metric: `mse` `mae` `rmse`'''
        Y_pred = self.predict(X)
        if metric == 'mse':
            return np.mean(np.square(Y - Y_pred))
        elif metric == 'mae':
            return np.mean(np.abs(Y - Y_pred))
        elif metric == 'rmse':
            return np.sqrt(np.mean(np.square(Y - Y_pred)))
        else:
            raise Exception('Unknown metric')

    def calc_accuracy(self, X, Y):
        Y_pred = self.predict(X)
        accuracy = np.sum(Y_pred == Y) / len(Y) * 100 
        return accuracy

    def chech_y_size(self,X,y):
        y_true = y
        y_pred = self.predict(X)
        return y_true, y_pred

    def plot_metrics(self, train_losses, test_losses, train_accuracies, test_accuracies):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.title('Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.title('Accuracy per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.tight_layout()
        plt.show()

