import tensorflow as tf
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

max_epochs=100
early_stop_epochs = 20
evaluate_every_n_step = 20
batch_size=32
keep_rate = 0.5

def get_input_fn(x,y, train=False, feature_name='neurons', batch_size=32):
  """
  Returns input function.
  """
  x = {feature_name: x}
  shuffle = True if train == True else False
  num_threads = 4 if train == True else 1
  num_epochs = None if train == True else 1

  input_fn = tf.estimator.inputs.numpy_input_fn(
    x, y, batch_size=batch_size, shuffle=shuffle,
    num_threads=num_threads, num_epochs=num_epochs)
  return input_fn

def mini_batches(x, y, shuffle=True):
  """
  Yields the data object with all properties sliced

  Args:
    data: numpy array
  """
  data_size = x.shape[0]
  num_batches = data_size//batch_size+(data_size%batch_size>0)
  indices = np.arange(0, data_size)
  if shuffle: np.random.shuffle(indices)
  for batch_num in range(num_batches):
    start_index = batch_num * batch_size
    end_index = min((batch_num + 1) * batch_size, data_size)
    new_indices = indices[start_index:end_index]
    yield x[new_indices], y[new_indices]

class RegModel():
  def __init__(self, x_dims, reg_scale):
    # l1_reg = tf.contrib.layers.l1_regularizer(scale=reg_scale, scope=None)
    self.x = tf.placeholder(tf.float32, shape=[None, x_dims])
    self.y = tf.placeholder(tf.float32, shape=[None, 1])
    self.keep_rate = tf.placeholder(tf.float32)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    units = 10
    h = tf.layers.dense(self.x, units, activation=tf.nn.relu)
    h = tf.nn.dropout(h,self.keep_rate)

    logits = tf.layers.dense(h, 1, activation=None)

    # Classification loss/prediction
    # logits = tf.matmul(h, W) + b
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=logits)
    self.predict = tf.round(tf.sigmoid(logits))

    # L1 regularization
    # weights = tf.trainable_variables()
    # reg_penalty = tf.contrib.layers.apply_regularization(l1_reg, weights)
    # reg_penalty = reg_scale*tf.reduce_sum(tf.abs(W))

    # Total Loss
    self.cost = tf.reduce_mean(loss) # loss for this batch

    # Optimization
    self.optimizer = tf.train.AdamOptimizer(0.001).minimize(\
                                            self.cost, global_step=global_step)

def evaluate(sess, model, x, y):
  x_size, x_dims = x.shape
  num_batches = x_size//batch_size+(x_size%batch_size>0)
  losses = np.zeros(num_batches) # store losses
  batch_weight = np.zeros(num_batches) # weights to balance av
  y_preds = np.zeros(y.shape)
  start_id = 0
  for batch_x, batch_y in mini_batches(x, y, shuffle=False):
    fetch = [model.predict]
    feed = {model.x: batch_x, model.keep_rate: 1}
    y_pred = sess.run(fetch, feed)[0]
    b_size_cur=batch_x.shape[0]
    y_preds[start_id:start_id+b_size_cur] = y_pred
    start_id += b_size_cur
  f1 = f1_score(y, y_preds, pos_label=1, average='binary')
  return f1

def log_regression_tf(trX, trY, vaX, vaY, teX=None, teY=None, penalty='l1',
    C=2**np.arange(-8, 1).astype(np.float), seed=42):
  """
  Logistic regression
  Args:
    C: numpy vector of inv of regularization strength (smaller values ->
      stronger regularization)
  """
  # Expand y dims
  trY = np.expand_dims(trY, axis=1)
  vaY = np.expand_dims(vaY, axis=1)
  teY = np.expand_dims(teY, axis=1)

  x_size, x_dims = trX.shape
  best_score = 0
  maybe_stop = 0
  num_batches = x_size//batch_size+(x_size%batch_size>0)
  with tf.Session() as sess:
    reg_scale = 0.125
    reg_scale = 0.01
    model = RegModel(x_dims, reg_scale)
    tf.global_variables_initializer().run()
    for epoch in range(max_epochs):
      print('train epoch: ',epoch, end='')
      losses = np.zeros(num_batches) # store losses
      batch_weight = np.zeros(num_batches) # weights to balance av
      i=0
      for batch_x, batch_y in mini_batches(trX, trY, shuffle=True):
        fetch = [model.optimizer, model.cost]
        feed = {model.x: batch_x, model.y: batch_y, model.keep_rate: keep_rate}
        _, loss = sess.run(fetch, feed)
        losses[i] = loss
        batch_weight[i] = batch_x.shape[0]/x_size
        i+=1
      av_loss = np.average(losses, weights=batch_weight)
      print(' | loss: ', av_loss, end='')
      f1_val = evaluate(sess, model, vaX, vaY)
      print(' | validation f1: ', f1_val)

      # Early stop check
      if f1_val >= best_score:
        maybe_stop = 0
        best_score = f1_val
        f1_test = evaluate(sess, model, teX, teY)
      else:
        maybe_stop += 1
      if maybe_stop >= early_stop_epochs:
        a = 1
        break

    print('val f1: ', best_score)
    print('test f1: ', f1_test)

    weights = tf.trainable_variables()[0]
    coefs = weights.eval()
    nnotzero = np.sum(coefs != 0)
    print('used weights: ', nnotzero)
    return f1_test, best_score, coefs

def log_regression_sklearn(trX, trY, vaX, vaY, teX=None, teY=None, penalty='l1',
    C=2**np.arange(-8, 1).astype(np.float), seed=42):
  """
  Logistic regression
  Args:
    C - numpy vector of inv of regularization strength (smaller values ->
        stronger regularization)
  """
  # Find best regularization coefficient
  scores = []
  for i, c in enumerate(C):
    # Model will overfit, we need to test after n steps
    model = LogisticRegression(C=c, penalty=penalty, random_state=seed+i)
    model.fit(trX, trY)
    y_pred = model.predict(vaX)
    f1_micro = f1_score(vaY, y_pred, pos_label=1, average='binary')
    score = model.score(vaX, vaY)
    scores.append(f1_micro)
  c = C[np.argmax(scores)]
  model = LogisticRegression(C=c, penalty=penalty, random_state=seed+len(C))
  model.fit(trX, trY)
  nnotzero = np.sum(model.coef_ != 0)
  if teX is not None and teY is not None:
    y_pred = model.predict(teX)
    f1_micro = f1_score(teY, y_pred, pos_label=1, average='binary')
    score = f1_micro*100
  else:
    y_pred = model.predict(vaX)
    f1_micro = f1_score(vaY, y_pred, pos_label=1, average='binary')
    score = f1_micro*100
  return score, c, nnotzero

