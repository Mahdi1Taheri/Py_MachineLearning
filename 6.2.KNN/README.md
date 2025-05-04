# Assignment 42, KNN on [ANSUR II](https://www.openlab.psu.edu/ansur2/) dataset
About Dataset: The Anthropometric Survey of US Army Personnel(ANSUR). dataset includes 93 measures for over 6,000 adult US military personnel (4,082 men and 1,986 women).<br>
You can download the dataset and read the info [here](https://www.openlab.psu.edu/ansur2/).<br>
in this Assignment, first of all, I improved my implemented KNN algorithm, after that I worked on the dataset, some works like preprocesses. here are the details:
## Preprocesses: weight, stature and gender datatype
* Weight unit converted from grams to kilograms.
* Stature unit converted from millimeters to centimeters.
* Gender datatype changed to 0 and 1.(0 for female, 1 for male)

## Height of men and women on plot
**A. Why is the data of men higher than the data of women?** Because men are taller than women<br>
**B. Why is the data of men more right than the data of women?** On average, men weigh more than women, and the more men's data is to the right in the graph, it means they weigh more.<br>
<br>
![height comparison](https://github.com/Mahdi1Taheri/Py_MachineLearning/blob/main/Assignment42/output/heights.png)
## Accuracy of KNN algorithm on ANSUR dataset
| k value     | Score  |
| :---   | :----: |
| 3      | 82.54%  |
| 5      | 83.53%  |
| 7      | 83.28%  |
| 9     | 83.77%  |
| 11     | 83.86%  |
| 13     | 84.35%  |
| 15     | 84.35%  |
## Confusion matrix
![confusion matrix](https://github.com/Mahdi1Taheri/Py_MachineLearning/blob/main/Assignment42/output/confusion_matrix.png)
