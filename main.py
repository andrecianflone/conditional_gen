import codecs
import json
import pickle
import numpy as np
from models import log_regression_tf, log_regression_sk
from helper import histogram, bar_chart, get_x_y
import matplotlib.pyplot as plt
import time

# Load pickled data

data = pickle.load(open("neurons.pickle", "rb"))
# ids is dictionary of disc IDs, discs is numpy array
ids, h_arr = data['ids'], data['discs']
h_size = h_arr.shape[1]


# Tensorflow log reg
# print('Training with tensorflow logistic regression')
# t1 = time.time()
# f1_test, f1_val, coefs = log_regression_tf(trX, trY, vaX, vaY, teX, teY)
# print('training min: {}'.format(round((time.time() - t1)/60, 1)))
# print('%05.2f test f1'%f1_test)
# print('%05.2f validation f1'%f1_val)

# Which relation?
relations = ['Temporal', 'Contingency', 'Expansion', 'Comparison']
folders = ['binary_implicit', 'binary_all_rel', 'binary_explicit']

def train(folder, relation):
  pass

plot_number = 1
for relation in relations:
  print('**Testing on relation: ', relation)
  for folder in folders:
    train_path = 'data/'+folder+'/one_v_all_train.json'
    val_path = 'data/'+folder+'/one_v_all_dev.json'
    test_path = 'data/'+folder+'/one_v_all_test.json'
    # Get data
    trX, trY =  get_x_y(ids, h_arr, train_path, relation)
    vaX, vaY =  get_x_y(ids, h_arr, val_path, relation)
    teX, teY =  get_x_y(ids, h_arr, test_path, relation)

    # SKlearn log reg
    print('Training with sklearn logistic regression')
    t1 = time.time()
    full_rep_acc, c, nnotzero, notzero_coefs_ids, notzero_coefs = \
                                log_regression_sk(trX, trY, vaX, vaY, teX, teY)
    print('training min: {}'.format(round((time.time() - t1)/60, 1)))
    print('%05.2f test accuracy'%full_rep_acc)
    print('%05.2f regularization coef'%c)
    print('%05d features used'%nnotzero)

    # Chart
    title = relation + ' ' + folder
    label = 'test acc {0:0.2f}, reg coef {1:0.2f}, features used: {2:0.0f}'.format(\
                                                        full_rep_acc, c, nnotzero)
    bar_chart(notzero_coefs, title, label, 4,3,plot_number)
    plot_number += 1
plt.show()


# bar_chart(notzero_coefs)
# histogram(notzero_coefs, bins=100)
print('goodbye world')
