from pprint import pprint
import codecs
import json
from sentiment_neuron.utils import sst_binary
from matplotlib import pyplot as plt
import pickle
import time
import numpy as np

def create_one_v_all_dataset(all_pdtb_path, hold_test, hold_val):
  """
  Creates a new dataset because:

  The problem with the PDTB is that it is hugely unbalanced, which is a problem
  when performing logistic regression + l1 reg. For the Temporal case, for eg,
  dataset is 85/961 pos/neg/. All coefs go to zero, we have no neuron info.

  Args:
    all_pdtb_path: a json of the entire PDTB dataset
    hold_test: percentage held out for test
    hold_val: percentage held out for val
  """
  pass

def get_x_y(ids, h_arr, path, relation=None):
  """
  Args:
    ids: a dictionary, key is discourse ID, value is arr index of 'h_arr'
    h_arr: numpy array of hidden unit values
  """
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

def get_discourse_data(path):
  """
  Args:
    path - data file
  Returns:
    disc_dic - maps discourseID to index of disc_text list
    disc_text - list of raw text
  """
  disc_dic = {}
  disc_text = []
  count = 0
  with codecs.open(path, encoding='utf8') as f:
    for line in f:
      j = json.loads(line)
      dID  = j['ID']
      arg1 = j['Arg1']['RawText']
      arg2 = j['Arg2']['RawText']

      # Concat the two
      text = "{} {}".format(arg1, arg2)
      disc_dic[dID] = count
      disc_text.append(text)
      count += 1

  return disc_dic, disc_text

def extract_neurons(ids, discs, save_path):
  """
  Get the hidden state params from the language model and save to disc
  with dictionary of id as pickled file
  """
  from sentiment_neuron.encoder import Model
  # get hidden unit values as npy
  model = Model()
  h = model.transform(discs)

  data = {'ids': ids, 'discs': h}
  pickle.dump(data, open(save_path, "wb"))
  print("saved pickle")

def histogram(arr, bins):
  plt.hist(arr, bins=bins, label='coef value')
  plt.legend()
  plt.show()

def bar_chart(arr, title, label, row, col, position):
  x = range(arr.shape[0])
  plt.subplot(row, col, position)
  plt.bar(x, arr, width=1)
  plt.title(title)
  plt.xlabel(label)

# def bar_chart(arr):
  # x = range(arr.shape[0])
  # fig = plt.figure()
  # ax = plt.subplot(111)
  # ax.bar(x, arr, width=1, color='b')
  # plt.show()

if __name__ == "__main__":
  # ids is dictionary of disc IDs, discs is list of raw text
  ids, discs = get_discourse_data('data/all_pdtb.json')
  extract_neurons(ids, discs, "neurons.pickle")

def get_unique_labels(file_path):
  """ Load dict """
  with codecs.open(file_path, encoding='utf-8') as f:
    dictionary = json.load(f)
  relations = set([rel for rel in dictionary.values()])
  return relations


