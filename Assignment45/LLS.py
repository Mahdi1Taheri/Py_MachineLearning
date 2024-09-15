import numpy as np

class LLS:
    def __init__(self):
        self.w = None 

    def fit(self, X_train, y_train):
        self.w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

    def predict(self, X_test):
        y_pred = X_test @ self.w
        return y_pred
    
    def evaluate(self, X_test, Y_test,metric):
        Y_pred = self.predict(X_test)
        error = Y_test - Y_pred

        if metric == 'mae':
            diff = Y_pred - Y_test
            abs_diff = np.absolute(diff)
            loss = abs_diff.mean()
        elif metric == 'mse':
            diff = Y_pred - Y_test
            differences_squared = diff ** 2
            loss = differences_squared.mean()
        elif metric == 'rmse':
            diff = Y_pred - Y_test
            differences_squared = diff ** 2
            mean_diff = differences_squared.mean()
            loss = np.sqrt(mean_diff)



        return loss
    

def my_train_test_split(X, y, test_size=0.2, shuffle=True, random_seed=None):
    """
    Splits the dataset into training and testing sets.

    Returns:
    X_train, X_test, y_train, y_test : array-like
        The training and testing splits for both features and target.
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)

    n_samples = len(X)
    
    # Shuffle the data
    if shuffle:
        indices = np.random.permutation(n_samples)
        X = np.array(X)[indices]
        y = np.array(y)[indices]
    
    # Calculate the size of the test set
    test_size = int(n_samples * test_size)
    
    # Split the data
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    
    return X_train, X_test, y_train, y_test

    
        