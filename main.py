import codecs
import json
import pickle
import numpy as np
from models import log_regression_tf, log_regression_sk
from helper import histogram, bar_chart, get_x_y, get_unique_labels
import matplotlib.pyplot as plt
import pickle
import time

# Load hidden states previously extracted
print("loading hidden states")
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

# Which relation? Use mapping file
relations = get_unique_labels('data/mapping_none.json')
folders = ['fine_binary_implicit', 'fine_binary_all', 'fine_binary_explicit']

# relations = ['Temporal', 'Contingency', 'Expansion', 'Comparison']
# folders = ['binary_implicit', 'binary_all_rel', 'binary_explicit']

def train(folder, relation):
  pass

def chart(pkl):
  """
  Chart list of results
  """
  # Loop through list

  # Chart
  title = relation + ' ' + folder
  label = 'test acc {0:0.2f}, reg coef {1:0.2f}, features used: {2:0.0f}'.format(\
                                                      full_rep_acc, c, nnotzero)
  bar_chart(notzero_coefs, title, label,len(relations),len(folder),plot_number)
  plt.show()

t0 = time.time()
plot_number = 0
results = [1]
for relation in relations:
  print('**Testing on relation: ', relation)
  for folder in folders:
    plot_number += 1
    print('computing logistic reg for {} in dir {}'.format(relation, folder))
    train_path = 'data/'+folder+'/train.json'
    val_path = 'data/'+folder+'/dev.json'
    test_path = 'data/'+folder+'/test.json'
    # Get data
    trX, trY =  get_x_y(ids, h_arr, train_path, relation)
    vaX, vaY =  get_x_y(ids, h_arr, val_path, relation)
    teX, teY =  get_x_y(ids, h_arr, test_path, relation)

    # Skip if no data
    if len(trX)==0 or len(vaX)==0:continue

    # SKlearn log reg
    print('Training with sklearn logistic regression')
    t1 = time.time()
    score_te, score_va, c, nnotzero, notzero_coefs_ids, notzero_coefs = \
                                log_regression_sk(trX, trY, vaX, vaY, teX, teY)
    print('training min: {}'.format(round((time.time() - t1)/60, 1)))
    print('%05.2f test accuracy'%score_te)
    print('%05.2f regularization coef'%c)
    print('%05d features used'%nnotzero)

    results.append({
        'relation'         : relation,
        'folder'           : folder,
        'score_test'       : score_te,
        'score_val'        : score_va,
        'size_train'       : len(trX),
        'size_val'         : len(vaX),
        'size_test'        : len(teX),
        'coefficient'      : c,
        'nnotzero'         : nnotzero,
        'notzero_coefs_ids' : notzero_coefs_ids,
        'notzero_coefs'    : notzero_coefs
        })

pickle.dump(results, open("results.pkl","wb"))
print('total exec time: {} min'.format(round((time.time() - t0)/60, 1)))

# bar_chart(notzero_coefs)
# histogram(notzero_coefs, bins=100)
print('goodbye world')
