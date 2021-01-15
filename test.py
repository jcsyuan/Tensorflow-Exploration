####################################################### IMPORTS #######################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf


####################################################### SETUP #######################################################

# load entire dataset
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
# load survived variable to another variable
y_train = dftrain.pop('survived') # training data
y_eval = dfeval.pop('survived') # testing data

# setup non-numerical columns
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
# setup numerical columns
NUMERIC_COLUMNS = ['age', 'fare']
# find all possible feature names of each feature column
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    # gets a list of all unique values from given feature column
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
# find all possible feature numbers of each numerical feature column
for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# function required for setting up training and testing, returns inner function (doesn't evaluate)
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    # function that is returned by outer function, loads data epoch number of times
    def input_function():
        # create tf.data.Dataset object with data and its label
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        # randomize order of data
        if shuffle:
            ds = ds.shuffle(1000)
        # split dataset into batches of 32 and load, then repeat process for number of epochs
        ds = ds.batch(batch_size).repeat(num_epochs)  
        # return a batch of the dataset
        return ds
    # return a function object for use
    return input_function


####################################################### TRAINING #######################################################

# create function objects that will return dataset objects to be fed into the model, one for training, one for testing
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# create a linear estimtor by passing the feature columns created earlier
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

# train
linear_est.train(train_input_fn)
# testing: get model metrics/stats by testing on testing data
result = linear_est.evaluate(eval_input_fn)

# print the result variable (a dict of stats about our model)
print(result['accuracy'])
pred_dicts = list(linear_est.predict(eval_input_fn))
# print graph of data
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
probs.plot(kind='hist', bins=20, title='predicted probabilities')