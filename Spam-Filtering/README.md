# Logistic Regression Spam Filter

Lets make a spam filter using logistic regression. We will classify messages to be either ham or spam. The dataset we’ll use is the [SMSSpamCollection dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection). The dataset contains messages, which are either spam or ham.

## what is logistic regression?

> Logistic regression is a simple classification algorithm. Given an example, we try to predict the probability that it belongs to “0” class or “1” class.

> Remember that with linear regression, we tried to predict the value of y(i) for x(i). Such continous output is not suited for the classification task.

Given the logisitic function and an example, it always returns a value between one and zero.


![Logistic Regression Model](https://raw.githubusercontent.com/hasansajedi/StartMachineLerning/master/images/logistic-function.png)

## Spam Filter Code
We load the dataset using pandas. Then we split in a training and test set. We extract text features known as TF-IDF features, because we need to work with numeric vectors.

Then we create the logistic regression object and train it with the data. Finally we create a set of messages to make predictions.

