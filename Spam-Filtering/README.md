logistic regression spam filter

Lets make a spam filter using logistic regression. We will classify messages to be either ham or spam. The dataset we’ll use is the SMSSpamCollection dataset. The dataset contains messages, which are either spam or ham.

spam filter code
We load the dataset using pandas. Then we split in a training and test set. We extract text features known as TF-IDF features, because we need to work with numeric vectors.

Then we create the logistic regression object and train it with the data. Finally we create a set of messages to make predictions.
