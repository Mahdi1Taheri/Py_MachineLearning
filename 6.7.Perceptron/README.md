# Perceptron (Assignment 6.7 / 47)
## Perceptron on [Surgical](https://www.kaggle.com/omnamahshivai/surgical-dataset-binary-classification) data
At first, I fit an object-oriented Perceptron algorithm on data.<br>
Perceptron class used on this data: `perceptron.py`
* Plotting Accuracy and Loss in each epoch<br><br>
  ![accuracy plot](https://github.com/Mahdi1Taheri/Py_MachineLearning/blob/main/6.7.Perceptron/output/loss_acc.png)<br>
* Confusion Matrix <br><br>
  ![cm](https://github.com/Mahdi1Taheri/Py_MachineLearning/blob/main/6.7.Perceptron/output/confusion_matrix.png)
## Perceptron on [Weather](https://github.com/Mahdi1Taheri/Py_MachineLearning/blob/main/6.7.Perceptron/input/weatherHistory.csv) data
in this part, I used a different Perceptron class named `Perceptron2.py` because the problem is NOT Linear.
### About Dataset
In this dataset The weather of a particular city has been recorded once an hour over the years, so there are 24 weather records per day.<br>
it starts from day 84 to day 304. 221 rows at all
### Average temperature per day
so, using these 24 temperatures per hour, I calculated the average temperature for the whole day.<br>
![average temperature](https://github.com/Mahdi1Taheri/Py_MachineLearning/blob/main/6.7.Perceptron/output/temp_per_day.png)<br>
average temp on chart:<br>
![avg temp chart](https://github.com/Mahdi1Taheri/Py_MachineLearning/blob/main/6.7.Perceptron/output/mean_temp_per_day.png)
### Training phase
using perceptron I trained a linear model on this data.
and write `evaluate()` function.<br>
* `evaluate()` result: `{'MSE': 53.0735135583146, 'MAE': 5.92119708975179}`
  * `MAE : 5` means that my model is on average 5°C different from the actual value. (around %12.5 error rate)
<br>
`predict2()`: this function takes X and day number as input and predicts the temperature on that day
