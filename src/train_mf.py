import pandas as pd
import numpy as np
import scipy.sparse as sparse
import time
from scipy.sparse.linalg import spsolve
from sklearn.model_selection import KFold


class Client:
  def __init__(self, item_rating, item_num, factor_num, alpha, l2_reg):
    self.item_rating = item_rating
    self.item_num = item_num
    self.factor_num = factor_num
    self.alpha = alpha
    self.l2_reg = l2_reg
  
  def get_rating_vector(self, implicit = True):
    # convert list of rating values to implicit rating vector
    self.rating_vector = np.zeros(self.item_num)
    for item in self.item_rating:
      if implicit:
        self.rating_vector[item-1] = 1
      else:
        self.rating_vector[item-1] = self.item_rating[item]
  
  def train_test_split(self, pct):
    # split into test set and train & validation set
    self.test_num = self.item_num - int(self.item_num * pct)
    self.train_val_vector = self.rating_vector.copy()
    self.test_index = np.random.choice(np.arange(self.item_num),size=self.test_num,replace=False)
    self.train_val_vector[self.test_index] = 0

  def train_test_split_no_val(self, pct):
    # split into test set and train set (no validation set)
    self.test_num = self.item_num - int(self.item_num * pct)
    self.train_vector = self.rating_vector.copy()
    self.test_index = np.random.choice(np.arange(self.item_num),size=self.test_num,replace=False)
    self.train_vector[self.test_index] = 0
    # self.c_vec = 1 + self.alpha * (self.train_vector != 0)
    self.c_vec = 1.0 * (self.train_vector != 0)
  
  def cross_val_split(self, test_pct, n_folds):
    # split train & validation set into k folds
    self.train_test_split(test_pct)
    all_index = np.arange(self.item_num)
    self.k_fold_vector = []
    kf = KFold(n_splits=n_folds)
    train_index = all_index[~np.in1d(all_index,self.test_index)]
    np.random.shuffle(train_index)
    for _, val_index in kf.split(train_index):
      self.k_fold_vector.append(list(train_index[val_index]))
  
  def train_val_split(self, fold_idx):
    # split the train & validation set into validation and train set
    self.train_vector = self.train_val_vector.copy()
    self.val_index = self.k_fold_vector[fold_idx]
    self.train_vector[self.val_index] = 0
    self.val_index = np.array(self.val_index)
    self.c_vec = 1 + self.alpha * (self.train_vector != 0)
  
  def initialize_user_lfm(self):
    # initialize user latent factor vector
    self.user_lfm = np.random.random(size=self.factor_num)

  def ALS_step(self, item_lfm):
    # update user latent vector
    c_matrix = sparse.diags(self.c_vec)
    Y = sparse.csr_matrix(item_lfm)
    p = sparse.csr_matrix(self.train_vector).T
    YCYT = Y.dot(c_matrix).dot(Y.T)
    YCp = Y.dot(c_matrix).dot(p)
    self.user_lfm = spsolve((YCYT + self.l2_reg * sparse.eye(self.factor_num)), YCp)

    # calculate client's gradient pass to server
    grad_to_server = []
    for i in range(self.item_num):
      c = self.c_vec[i]
      p = self.train_vector[i]
      y = item_lfm[:,i]
      xTy = self.user_lfm.dot(y)
      grad = c * (p - xTy) * self.user_lfm
      grad_to_server.append(grad)
    grad_to_server = np.array(grad_to_server).T
    return grad_to_server
  
  def predict_rating(self, item_lfm):
    # predict the rating for each client
    self.rating_pred = self.user_lfm.dot(item_lfm)
  
  def cal_val_loss(self):
    # calculate the validation precision and recall on each client
    # return to server to compute the aggregate precision, recall and F1
    val_actual = self.rating_vector[self.val_index]
    val_pred = self.rating_pred[self.val_index]
    recommendation_list = val_pred.argsort()[-10:][::-1]
    rated_item_list = np.where(val_actual != 0)[0]
    intersect_item_list = np.intersect1d(rated_item_list, recommendation_list)
    precision = len(intersect_item_list) / recom_num
    try:
      recall = len(intersect_item_list) / len(rated_item_list)
    except ZeroDivisionError:
      recall = 0
    return precision, recall

  def cal_train_loss(self):
    # compute part of loss function on client side
    sum_square_error = self.c_vec.dot((self.train_vector - self.rating_pred) ** 2)
    user_l2_reg = sum(self.user_lfm ** 2)
    loss = sum_square_error + user_l2_reg
    return loss
  
  def cal_average_precision(self, rated_item_list, recommendation_list):
    # calculate average precision for each client
    recom_num = len(recommendation_list)
    rated_num = len(rated_item_list)
    accurate_num = 0
    average_precision = 0
    for i in range(recom_num):
        rel = int(recommendation_list[i] in rated_item_list)
        accurate_num += rel
        current_precision = accurate_num / (i + 1)
        average_precision += current_precision * rel
    if rated_num > 0:
        average_precision /= rated_num
    return average_precision

  def cal_test_accuracy(self, recom_num = 10):
    # compute part of test accuracy on client side
    test_actual = self.rating_vector[self.test_index]
    test_pred = self.rating_pred[self.test_index]
    non_zero_test = (test_actual>0)
    n_non_zeros = non_zero_test.sum()
    if n_non_zeros > 0:
        non_zero_pred = test_pred[test_actual>0]
        non_zero_pred = np.clip(non_zero_pred, 1, 5)
        rmse = np.sqrt(sum((test_actual[non_zero_test] - non_zero_pred) ** 2) / n_non_zeros)
    else:
        rmse = None

    recommendation_list = test_pred.argsort()[-10:][::-1]
    rated_item_list = np.where(test_actual != 0)[0]
    intersect_item_list = np.intersect1d(recommendation_list, rated_item_list)
    precision = len(intersect_item_list) / recom_num
    try:
      recall = len(intersect_item_list) / len(rated_item_list)
    except ZeroDivisionError:
      recall = 0
    avg_precision = self.cal_average_precision(rated_item_list, recommendation_list)
    return rmse, precision, recall, avg_precision

class Server:
  def __init__(self, item_num, factor_num, l2_reg):
    self.item_num = item_num
    self.factor_num = factor_num
    self.l2_reg = l2_reg

  def initialize_item_lfm(self):
    # initialize item latent factor matrix
    self.item_lfm = np.random.random(size=(self.factor_num,self.item_num))
  
  def calculate_gradient(self, agg_grad):
    # calculate gradient for item matrix
    # agg_grad: sum of gradient from client side
    self.item_gradient = -2 * agg_grad + 2 * self.l2_reg * self.item_lfm
    return self.item_gradient
  
  def gradient_descend(self, lr):
    # normal SGD update
    self.item_lfm = self.item_lfm - lr * self.item_gradient

  def gradient_descend_Adam(self, lr, m_adjusted, v_adjusted, esp = 10-8):
    # Adam update
    self.item_lfm = self.item_lfm - lr / (np.sqrt(v_adjusted) + esp) * m_adjusted

  def cal_train_loss(self, agg_train_loss):
    # compute total train loss
    item_l2_reg = (self.item_lfm ** 2).sum()
    total_train_loss = agg_train_loss + item_l2_reg
    return total_train_loss

def validation():
  # compute f1 score on validation set
  precision = 0
  recall = 0
  item_lfm = server.item_lfm
  for client in client_list:
    client.predict_rating(item_lfm)
    current_precision, current_recall = client.cal_val_loss()
    precision += current_precision
    recall += current_recall
  precision /= user_num
  recall /= user_num
  f1 = 2 * precision * recall / (precision + recall)
  return f1

def test():
  # calculate accuracy on test set
  rmse = []
  precision = 0
  recall = 0
  _map = 0
  item_lfm = server.item_lfm
  for client in client_list:
    client.predict_rating(item_lfm)
    current_rmse, current_precision, current_recall, current_ap = client.cal_test_accuracy()
    if current_rmse is not None:
        rmse.append(current_rmse)
    precision += current_precision
    recall += current_recall
    _map += current_ap
  rmse = np.mean(rmse)
  precision /= user_num
  recall /= user_num
  _map /= user_num
  f1 = 2 * precision * recall / (precision + recall)
  print('RMSE:', rmse)
  print('Precision:', precision)
  print('Recall:', recall)
  print('MAP:', _map)
  print('F1:', f1)
  return rmse, precision, recall, f1, _map

def train(epoch, n_iter, lr, adam = False, beta1 = 0.4, beta2 = 0.99, esp = 10-8):
  np.random.shuffle(client_list)
  server.initialize_item_lfm()
  item_lfm = server.item_lfm
  n_per_epoch = int(user_num / n_iter)
  m = 0
  v = 0
  best_rmse = 100
  for e in range(epoch):
    for i in range(n_iter):
      u_start = i * n_per_epoch
      if i == n_iter - 1:
        u_end = user_num
      else:
        u_end = (i + 1) * n_per_epoch
      # update user latent factor and obtain aggregate gradient
      agg_grad = np.zeros((n_factors, item_num))
      for u in range(u_start, u_end):
        client = client_list[u]
        grad_to_server = client.ALS_step(item_lfm)
        agg_grad += grad_to_server

      agg_grad = agg_grad / (u_end - u_start) * user_num

      # update item latent factor on server by Adam method
      if adam == True:
        grad = server.calculate_gradient(agg_grad)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_adjusted = m / (1 - beta1)
        v_adjusted = v / (1 - beta2)
        server.gradient_descend_Adam(lr, m_adjusted, v_adjusted, esp)
      # update item latent factor by normal SGD
      else:
        server.calculate_gradient(agg_grad, l2_reg)
        server.gradient_descend(lr)

      # compute train loss for current iteration
      item_lfm = server.item_lfm
      agg_train_loss = 0
      for u in range(user_num):
        client = client_list[u]
        client.predict_rating(item_lfm)
        loss = client.cal_train_loss()
        agg_train_loss += loss
      total_train_loss = server.cal_train_loss(agg_train_loss)
      print('Epoch {}, Iter {}, total loss {}'.format(e+1, i+1, total_train_loss))
      rmse, precision, recall, f1, _map = test()
      if rmse < best_rmse:
        best_rmse = rmse
  print("best rmse:", best_rmse)

if __name__ == "__main__":
    rating_path = "/opt/data/private/EfficientRecsys/data/ml-1m/ratings.dat"
    rating = pd.read_csv(rating_path, delimiter='::', header=None, names=["UserID", "ItemID", "Rating", "Timestamp"])
    item_num = rating.ItemID.max()
    user_num = rating.UserID.max()
    rating = rating.groupby('UserID').apply(lambda x: dict(zip(x.ItemID.to_list(), x.Rating.to_list())))

    # n_factors = 8
    l2_reg = 2
    alpha = 10000
    lr = 0.05
    n_iter = 10
    epoch = 50

    for n_factors in [6, 10, 16, 32, 64]:
        print("Number of latent factors:", n_factors)
        # initialize clients
        client_list = []
        for u in range(1, user_num+1):
            current_client = Client(rating[u], item_num, n_factors, l2_reg, alpha)
            current_client.get_rating_vector(implicit = False)
            current_client.train_test_split_no_val(0.8)
            current_client.initialize_user_lfm()
            client_list.append(current_client)

        # initialize server
        server = Server(item_num, n_factors, l2_reg)

        train(epoch, n_iter, lr, adam=True)