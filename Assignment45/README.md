# Assignment 45, processes and LLS on 2 datasets
## Tehran House Price dataset
first some preprocesses done on [this dataset](https://www.kaggle.com/code/soheiltehranipour/tehran-divar-ir-house-price-prediction):
* Dollar price updated to July 2023
* Removed incompleted datas
### Address of 5 most expensive houses in Tehran
![top expensive tehran house](https://github.com/Mahdi1Taheri/Py_MachineLearning/blob/main/Assignment45/output/top_5_expensive_teh.png)
### Evaluate model on training dataset
```
MAE: 2743178348.0
MSE: 2.3931261151317463e+19
RMSE: 4891958826.0
 ```
## Dollar Rial price dataset
Preprocesses on this [Dataset](https://github.com/M-Taghizadeh/Dollar_Rial_Price_Dataset):
* Dataset divided to "Ahmadinejad", "Rouhani" and "Raisi" Presidency using Date column
* "Date" column converted to pandas datetime
* Added Date_int for using in `train_test_split` scikit-learn function. (Date column changed to days count)
* "Close" column converted to int datatype
### Highest Dollar price in 3 Presidencies
![highest dollar price for each president](https://github.com/Mahdi1Taheri/Py_MachineLearning/blob/main/Assignment45/output/highest_dollar_per_president.png)
### Lowest Dollar price in 3 Presidencies
![Lowest dollar price for each president](https://github.com/Mahdi1Taheri/Py_MachineLearning/blob/main/Assignment45/output/lowest_dollar_per_president.png)
### Evaluate each model using MAE loss function
```
Ahmadinejad MAE: 2878.600518073426
Rouhani MAE: 31583.619981543423
Raisi MAE: 31298.935725282838
```
## fluctuations of Dollar Charts
### Dollar fluctuations at Ahmadinejad's presidency 
![Ahmadinejad](https://github.com/Mahdi1Taheri/Py_MachineLearning/blob/main/Assignment45/output/dollar_ahamdinejad.png)
### Dollar fluctuations at Rouhani's presidency 
![Ahmadinejad](https://github.com/Mahdi1Taheri/Py_MachineLearning/blob/main/Assignment45/output/dollar_rouhani.png)
### Dollar fluctuations at Raisi's presidency 
![Ahmadinejad](https://github.com/Mahdi1Taheri/Py_MachineLearning/blob/main/Assignment45/output/dollarRaisi.png)
### Comparison
![](https://github.com/Mahdi1Taheri/Py_MachineLearning/blob/main/Assignment45/output/dollar_comparison.png)
