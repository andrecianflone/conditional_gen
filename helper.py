from pprint import pprint
import codecs
import json
from utils import sst_binary
from sentiment_neuron.model.encoder import Model
import pickle
import time

model = Model()

def extract_neurons(path):
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

if __name__ == "__main__":
  # ids is dictionary of disc IDs, discs is list of raw text
  ids, discs = extract_neurons('data/all_pdtb.json')
  # get hidden unit values as npy
  h = model.transform(discs)

  data = {'ids': ids, 'discs': h}
  pickle.dump(data, open("neurons.pickle", "wb"))
  print("saved pickle")

