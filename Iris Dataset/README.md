# StartMachineLerning - ( Iris Dataset )

![StartMachineLerning with Hasan Sajedi](https://raw.githubusercontent.com/hasansajedi/StartMachineLerning/master/images/score.png)


>Requirements library:
- Python 3.6
- scikit-learn
- numpy
- pandas
- matplotlib

Use scikit-learn library for Machine Learning in python that's include all of you need to ML doing. for more information [visit site](http://scikit-learn.org/stable/).

> Result of code:

**SVC**
  - Result: [1.         1.         1.         1.         1.         1. 0.91666667 1.         1.         1.        ]
  - Mean: 0.9916666666666666
  - Standard Deviation: 0.025000000000000012

**KNN**
  - Result: [1.         1.         1.         1.         1.         1. 0.91666667 1.         1.         0.91666667]
  - Mean: 0.9833333333333332
  - Standard Deviation: 0.03333333333333335

**LoR**
  - Result: [1.         1.         0.91666667 1.         1.         1. 0.91666667 0.91666667 0.91666667 1.        ]
  - Mean: 0.9666666666666666
  - Standard Deviation: 0.04082482904638632

**LDA**
  - Result: [1.         1.         1.         0.91666667 1.         1. 0.91666667 1.         0.91666667 1.        ]
  - Mean: 0.975
  - Standard Deviation: 0.03818813079129868

**QDA**
  - Result: [1.         1.         1.         1.         1.         1. 0.91666667 1.         0.91666667 0.91666667]
  - Mean: 0.9749999999999999
  - Standard Deviation: 0.03818813079129868

**GNB**
  - Result: [1.         1.         1.         0.91666667 1.         1. 1.         1.         0.83333333 1.        ]
  - Mean: 0.975
  - Standard Deviation: 0.053359368645273735

**DT**
  - Result: [1.         1.         1.         0.91666667 1.         1. 0.91666667 1.         0.91666667 1.        ]
  - Mean: 0.975
  - Standard Deviation: 0.03818813079129868

**RF**
  - Result: [0.91666667 1.         1.         0.91666667 1.         1. 0.91666667 1.         0.91666667 1.        ]
  - Mean: 0.9666666666666666
  - Standard Deviation: 0.04082482904638632


INFO:Best model is (**SVC**) and model score is (0.025000000000000012)

INFO:SVC score is: 93.333333%

INFO:Confusion matrix is: 

[[ 7  0  0]
 [ 0 10  2]
 [ 0  0 11]]

INFO:Classification report is:                  

|                  |      precision  |   recall | f1-score   | support  |
| ---------------- | --------------- | -------- | ---------- | -------- |
|     Iris-setosa  |        1.00     |     1.00 | 1.00       | 7        |
| Iris-versicolor  |        1.00     |     0.83 | 0.91       | 12       |
|  Iris-virginica  |        0.85     |     1.00 | 0.92       | 11       |
|     avg / total  |       0.94      |     0.93 | 0.93       | 30       |

If this source code help you please :+1: this repository.
