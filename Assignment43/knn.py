import numpy as np
from collections import Counter
from PIL import Image
import cv2

class KNN:
    def __init__(self, k):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)
    
    def _predict(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        accuracy = np.sum(Y_pred == Y) / len(Y)
        return accuracy

class FindingNemo:
    def __init__(self, train_image):
        self.low_orange = np.array([1,190,10])
        self.high_orange = np.array([50,255,255])

        self.low_white = np.array([0, 0, 200])
        self.high_white = np.array([179, 55, 255])

        self.low_black = np.array([0, 0, 0])
        self.high_black = np.array([179, 255, 50])
        
        self.knn = KNN(k=3)
        X_train, Y_train = self.convert_image_to_dataset(train_image)
        self.knn.fit(X_train, Y_train)

        
    
    def convert_image_to_dataset(self, image):
        img_hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        pixels_lst_hsv = img_hsv.reshape(-1,3)

        mask_orange = cv2.inRange(img_hsv,self.low_orange,self.high_orange)
        mask_white = cv2.inRange(img_hsv,self.low_white,self.high_white)
        mask_black = cv2.inRange(img_hsv,self.low_black,self.high_black)

        final_mask = mask_orange + mask_white + mask_black

        Y_train = final_mask.reshape(-1,) // 255
        X_train = pixels_lst_hsv / 255

        return X_train, Y_train


    def remove_background(self, test_image):
        test_image = cv2.resize(test_image, (0,0), fx=.25,fy=.25)
        test_image = cv2.cvtColor(test_image,cv2.COLOR_BGR2RGB)
        test_image_hsv = cv2.cvtColor(test_image,cv2.COLOR_RGB2HSV)

        X_test = test_image_hsv.reshape(-1,3) / 255
        Y_pred = self.knn.predict(X_test)
        output = Y_pred.reshape(test_image.shape[:2])
        final_res = cv2.bitwise_and(test_image,test_image,mask=output)

        return final_res
    
class FindingDory:
    def __init__(self, train_image):
        self.low_blue = np.array([100, 150, 50])
        self.high_blue = np.array([140,255,255])

        self.low_yellow = np.array([30, 150, 150])
        self.high_yellow = np.array([50, 255, 255])

        self.low_black = np.array([0, 0, 0])
        self.high_black = np.array([179, 255, 35])
        
        self.knn = KNN(k=3)
        X_train, Y_train = self.convert_image_to_dataset(train_image)
        self.knn.fit(X_train, Y_train)

        
    
    def convert_image_to_dataset(self, image):
        image = cv2.resize(image, (0,0), fx=.25,fy=.25)
        img_hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        pixels_lst_hsv = img_hsv.reshape(-1,3)
        
        mask_blue = cv2.inRange(img_hsv,self.low_blue,self.high_blue)
        mask_yellow = cv2.inRange(img_hsv,self.low_yellow,self.high_yellow)
        mask_black = cv2.inRange(img_hsv,self.low_black,self.high_black)

        final_mask = mask_blue + mask_yellow + mask_black

        Y_train = final_mask.reshape(-1,) // 255
        X_train = pixels_lst_hsv / 255

        return X_train, Y_train


    def remove_background(self, test_image):
        test_image = cv2.cvtColor(test_image,cv2.COLOR_BGR2RGB)
        test_image_hsv = cv2.cvtColor(test_image,cv2.COLOR_RGB2HSV)

        X_test = test_image_hsv.reshape(-1,3) / 255
        Y_pred = self.knn.predict(X_test)
        output = Y_pred.reshape(test_image.shape[:2])
        final_res = cv2.bitwise_and(test_image,test_image,mask=output)

        return final_res