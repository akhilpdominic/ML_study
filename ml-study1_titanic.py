
import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as feature_column
import tensorflow as tf


dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train=dftrain.pop('survived')
y_test=dfeval.pop('survived')


dftrain.head()

dfeval

dftrain.describe()

dftrain.shape

dftrain['age'].hist(bins=20)

dfeval.shape

categorical_columns=[]
for i in dfeval.columns:
  categorical_columns.append(i)

categorical_columns.remove('fare')
categorical_columns.remove('age')

categorical_columns

numeric_columns=['fare','age']


feature_columns=[]
for feature_name in categorical_columns:
  voc=dftrain[feature_name].unique()
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name,voc))

for feature_name in numeric_columns:
  feature_columns.append(tf.feature_column.numeric_column(feature_name,dtype=tf.float32))
  


print(feature_columns)

def make_inp_fun(data_df,label_df,num_epochs=50,shuffle=True,batch_size=32):
  def inp_fun():
    ds=tf.data.Dataset.from_tensor_slices((dict(data_df),label_df))
    if shuffle:
      ds=ds.shuffle(1000)
    ds=ds.batch(batch_size).repeat(num_epochs)
    return ds
  return inp_fun

train_inp_fn=make_inp_fun(dftrain,y_train)
eval_inp_fn=make_inp_fun(dfeval,y_test,num_epochs=1)


linear_est=tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est=tf.estimator.LinearClassifier(feature_columns=feature_columns)
train_inp_fn=make_inp_fun(dftrain,y_train)
eval_inp_fn=make_inp_fun(dfeval,y_test,num_epochs=1,shuffle=False)


linear_est.train(train_inp_fn)
result=linear_est.evaluate(eval_inp_fn)


print(result)

pred_dicts=list(linear_est.predict(eval_inp_fn))

print(pred_dicts[0]['probabilities'][1])





