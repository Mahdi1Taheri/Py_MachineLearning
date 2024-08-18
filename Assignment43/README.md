# Working with KNN and sklearn datasets
## Finding Nemo
first I created a training dataset for nemo pixels and fitted my implemented KNN on train data. after that predicted Nemo pixels on the test image. <br>
then I put all the steps into a class and called it FindingNemo.<br>
| Original Image | Nemo predicted mask             |  Nemo Result |
:-------------------------:|:-------------------------:|:-------------------------:
![nemo original](https://github.com/Mahdi1Taheri/Py_MachineLearning/blob/main/Assignment43/input/dashe_nemo2.png) | ![nemo mask](https://github.com/Mahdi1Taheri/Py_MachineLearning/blob/main/Assignment43/output/nemo_predicted_mask.png) | ![nemo result](https://github.com/Mahdi1Taheri/Py_MachineLearning/blob/main/Assignment43/output/nemo_predicted_res.png)

## Finding Dory
Steps are the same as finding nemo steps but pixel colors are different.<br>
| Original Image | Dory predicted mask             |  Dory Result |
:-------------------------:|:-------------------------:|:-------------------------:
![dory original](https://github.com/Mahdi1Taheri/Py_MachineLearning/blob/main/Assignment43/input/blue_tang1.jpg) | ![dory mask](https://github.com/Mahdi1Taheri/Py_MachineLearning/blob/main/Assignment43/output/dory_mask.png) | ![dory result](https://github.com/Mahdi1Taheri/Py_MachineLearning/blob/main/Assignment43/output/dory_result.png)
 ## Iris Dataset
 Iris is a sklearn dataset. I evaluated my KNN algorithm on test dataset with different values of k, and here are the results:<br>
| K Value | Accuracy |
| ------------- | ------------- |
| 3  | 96.67% |
| 5  | 96.67%  |
| 7  | 96.67%  |
| 9  | 96.67%  |
| 11 | 96.67%  |
| 13  | 96.67%  |
<br>
Confusion matrix:
![](https://github.com/Mahdi1Taheri/Py_MachineLearning/blob/main/Assignment43/output/confusion_matrix_iris.png)

## Breast cancer dataset
steps are the same as iris dataset steps.
Accuracy results:<br>
| K Value | Accuracy |
| ------------- | ------------- |
| 3  | 93.01% |
| 5  | 94.41%  |
| 7  | 94.41%  |
| 9  | 93.71%  |
| 11 | 94.41%  |
| 13  | 94.41% |
<br>
Confusion matrix:<br>
![cm breast cancer](https://github.com/Mahdi1Taheri/Py_MachineLearning/blob/main/Assignment43/output/cm_breast_cancer.png)








