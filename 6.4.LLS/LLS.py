import numpy as np

class LLS:
    def __init__(self):
        self.w = None 

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X] 
        self.w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X] 
        y_pred = X_b @ self.w
        return y_pred
    
    
        