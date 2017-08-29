import codecs
import json
import pickle
import numpy as np
from sentiment_neuron.utils import sst_binary
from sentiment_neuron.utils import train_with_reg_cv as log_regression_sk
from models import log_regression_tf
import time

# Load pickled data

data = pickle.load(open("neurons.pickle", "rb"))
# ids is dictionary of disc IDs, discs is numpy array
ids, h_arr = data['ids'], data['discs']
h_size = h_arr.shape[1]

def get_x_y(path, relation=None):
  map_cls = {'positive': 1, 'negative': 0}
  x_size = sum(1 for line in open(path))
  x = []
  y = []
  row = 0
  with codecs.open(path, encoding='utf8') as f:
    for line in f:
      j = json.loads(line)

      # Check if wanted relation
      if j['Relation'] != relation: continue

      dID  = j['ID']
      y_sample = map_cls[j['Class']]

      # Get the row of the hidden units value
      x_sample = h_arr[ids[dID]]
      x.append(x_sample)
      y.append(y_sample)
  x = np.array(x)
  y = np.array(y)
  return x, y

relation = 'Contingency'
trX, trY =  get_x_y('data/one_v_all_train.json', relation)
vaX, vaY =  get_x_y('data/one_v_all_dev.json', relation)
teX, teY =  get_x_y('data/one_v_all_test.json', relation)

# Tensorflow log reg
print('Training with tensorflow logistic regression')
t1 = time.time()
f1_test, f1_val, coefs = log_regression_tf(trX, trY, vaX, vaY, teX, teY)
print('training min: {}'.format(round((time.time() - t1)/60, 1)))
print('%05.2f test f1'%f1_test)
print('%05.2f validation f1'%f1_val)

# SKlearn log reg
print('Training with sklearn logistic regression')
t1 = time.time()
full_rep_acc, c, nnotzero = log_regression_sk(trX, trY, vaX, vaY, teX, teY)
print('training min: {}'.format(round((time.time() - t1)/60, 1)))
print('%05.2f test accuracy'%full_rep_acc)
print('%05.2f regularization coef'%c)
print('%05d features used'%nnotzero)

