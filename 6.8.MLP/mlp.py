import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from tqdm import tqdm

class MLP:
    def __init__(self, lr, epochs, input_size, hidden_sizes, output_size):
        self.lr = lr
        self.epochs = epochs
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size 
        self._initialize()

    def _initialize(self):
        # weights
        self.w1 = np.random.randn(self.input_size, self.hidden_sizes[0])
        self.w2 = np.random.randn(self.hidden_sizes[0],self.hidden_sizes[1])
        self.w3 = np.random.randn(self.hidden_sizes[1], self.output_size)

        # bias
        self.B1 = np.random.randn(1, self.hidden_sizes[0])
        self.B2 = np.random.randn(1, self.hidden_sizes[1])
        self.B3 = np.random.randn(1, self.output_size)

    def activation(self, X, function='sigmoid'):
        if function == 'sigmoid':
            return 1 / (1 + np.exp(-X))
        elif function == 'softmax':
            return np.exp(X) / np.sum(np.exp(X))
        else:
            raise ValueError("Incorrect function")
    
    def forward(self, x):
        x = x.reshape(-1, 1)
        # forward 
        ## Layer 1
        self.layer1_out = self.activation(x.T @ self.w1 + self.B1)
        ## Layer 2
        self.layer2_out = self.activation(self.layer1_out @ self.w2 + self.B2)
        ## Layer 3
        self.layer3_out = self.activation(self.layer2_out @ self.w3 + self.B3, 'softmax') 
        self.y_pred = self.layer3_out
        return self.y_pred

    def back_propagation(self, x, y):
        x = x.reshape(-1, 1)
         # backpropagation - باید مشتق بگیریم
        ## Layer 3
        error = -2 * (y - self.y_pred)
        grad_B3 = error
        grad_w3 = error * self.layer2_out.T
        ## Layer 2
        error = error @ self.w3.T * self.layer2_out * (1 - self.layer2_out)
        grad_B2 = error
        grad_w2 = self.layer1_out.T @ error 
        ## Layer 1
        error = error @ self.w2.T * self.layer1_out * (1 - self.layer1_out)
        grad_B1 = error
        grad_w1 = x @ error  

        # update
        ## Layer 1
        self.w1 -= self.lr * grad_w1
        self.B1 -= self.lr * grad_B1
        ## Layer 3
        self.w2 -= self.lr * grad_w2
        self.B2 -= self.lr * grad_B2
        ## Layer 3
        self.w3 -= self.lr * grad_w3
        self.B3 -= self.lr * grad_B3

    
    def train(self, X_train, y_train):
        losses = []
        accuracies = []
        for epoch in range(self.epochs):
            for x, y in zip(X_train, y_train):
                x = x.reshape(1, -1)
                self.forward(x)
                self.back_propagation(x, y)
            accuracy, loss = self.evaluate(X_train, y_train)
            losses.append(loss)
            accuracies.append(accuracy)

        self.plot_metrics(losses, accuracies)
        
    def plot_metrics(self, losses, accuracies):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(losses, label='Train Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(accuracies, label='Test Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def predict(self, X):
        predictions = []
        for x in X:
            x = x.reshape(1, -1)
            output = self.forward(x)
            predictions.append(output)
        return np.array(predictions).reshape(-1, self.output_size)

    
    def evaluate(self, X_test, y_test):
        self.y_pred = self.predict(X_test)
        self.y_pred_labels = np.argmax(self.y_pred, axis=1)
        self.y_test_labels = np.argmax(y_test, axis=1)
        accuracy = np.sum(self.y_test_labels == self.y_pred_labels) / len(self.y_test_labels)
        loss = self.calc_loss(X_test, y_test)

        return accuracy, loss

    def precision_recall(self, y_test, num_classes):
        self.y_pred_labels = np.argmax(self.y_pred, axis=1)
        self.y_test_labels = np.argmax(y_test, axis=1)
        precisions = []
        recalls = []
        for cls in range(num_classes):
            TP = sum((self.y_test_labels == cls) & (self.y_pred_labels == cls))
            FP = sum((self.y_test_labels != cls) & (self.y_pred_labels == cls))
            FN = sum((self.y_test_labels == cls) & (self.y_pred_labels != cls))
            
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)

        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        
        return {'Precision': avg_precision, 'Recall': avg_recall}

    def calc_loss(self, X, Y, metric='rmse'):
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
        

    
    