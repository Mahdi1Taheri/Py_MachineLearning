# MLP (Multi-Layer Perceptron)
## Precision and Recall functions implementation
in this part, I implemented Precision and Recall functions in one function called `precision_recall` from scratch.<br><br>
My `precision_recall` function results: `Precision: 0.9393939393939394, Recall: 0.8857142857142857`<br>
SkLearn's `precision_score` and `recall_score` results: `Precision: 0.9424267997497738, Recall: 0.9413497306956609`
## Object-Oriented MLP
I also implemented an object-oriented MLP class from scratch in [mlp.py](https://github.com/Mahdi1Taheri/Py_MachineLearning/blob/main/6.8.MLP/mlp.py).<br>
you can see model training results in [model_test.ipynb](https://github.com/Mahdi1Taheri/Py_MachineLearning/blob/main/6.8.MLP/model_test.ipynb)
## OneHot Encoder and Decoder
I implemented two functions, one for onehot encoding called `onehot_encode()` and one for onehot decoding called `onehot_decode()`.
## Loss and Accuracy
model loss and accuracy results on train and test data:<br>

|  Train loss  | test loss |
| ------------- | ------------- |
| 0.05  | 0.12  |

|  Train Accuracy  | Test Accuracy |
| ------------- | ------------- |
| 0.98  | 0.89  |
### Plotted Result:
![](https://github.com/Mahdi1Taheri/Py_MachineLearning/blob/main/6.8.MLP/output/mlp_loss_acc.png)

