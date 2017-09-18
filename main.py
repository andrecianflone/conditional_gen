import codecs
import json
import pickle
import numpy as np
from helper import histogram, bar_chart, get_x_y, get_unique_labels
import matplotlib.pyplot as plt
import pickle
import time

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
  from models import log_regression_tf, log_regression_sk
  # Load hidden states previously extracted
  print("loading hidden states")
  data = pickle.load(open("neurons.pkl", "rb"))
  # ids is dictionary of disc IDs, discs is numpy array
  ids, h_arr = data['ids'], data['discs']
  h_size = h_arr.shape[1]

  t0 = time.time()
  plot_number = 0
  results = []
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

  # Save results
  pickle.dump(results, open("results.pkl","wb"))
  print('total exec time: {} min'.format(round((time.time() - t0)/60, 1)))

def find_result(data, relation, folder):
  for result in data:
    if result['relation'] == relation:
      if result['folder'] == folder:
        return result
  # If not found, return none
  return None

def chart(pkl, max_rows):
  """
  Chart list of results
  Args:
    pkl: file path to pickled results
    max_rows: max num of rows per chart
  """
  data = pickle.load(open(pkl, "rb"))
  relations = list(set([x['relation'] for x in data])).sort()
  folders   = list(set([x['folder'] for x in data])).sort()
  plots = max_rows*len(folders)

  # Loop through list
  plot_number = 0
  for relation in relations:
    for folder in folders:
      plot_number += 1
      # Get single result
      d = find_result(data, relation, folder)
      if d is None: continue
      # Chart
      title = relation + ' ' + folder
      label = 'size tr/va/te {:0.0f}/{:0.0f}/{:0.0f}, acc {:0.2f},\
        reg coef {:0.2f}, neurons: {:0.0f}'.format(d['size_train'],
            d['size_val'], d['size_test'], d['score_test'], d['coefficient'],
                                                                d['nnotzero'])
      bar_chart(d['notzero_coefs'], title, label, max_rows,\
                                                      len(folders), plot_number)
      if plot_number%plots==0:
        plt.show()
        plot_number=0

# histogram(notzero_coefs, bins=100)

if __name__=="__main__":
  chart("results.pkl", 4)

