import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[1]
  num_class = dW.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  loss = 0.0
  for i in xrange(num_train):
    X_i =  X[:,i]
    score_i = W.dot(X_i)
    stability = -score_i.max()
    exp_score_i = np.exp(score_i+stability)
    exp_score_total_i = np.sum(exp_score_i , axis = 0)
    for j in xrange(num_class):
      if j == y[i]:
        dW[j,:] += -X_i.T + (exp_score_i[j] / exp_score_total_i) * X_i.T
      else:
        dW[j,:] += (exp_score_i[j] / exp_score_total_i) * X_i.T
    numerator = np.exp(score_i[y[i]]+stability)
    denom = np.sum(np.exp(score_i+stability),axis = 0)
    loss += -np.log(numerator / float(denom))


  loss = loss / float(num_train) + 0.5 * reg * np.sum(W*W)
  dW = dW / float(num_train) + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[1]
  num_class = W.shape[0]
  dW = np.zeros(W.shape) # initialize the gradient as zero 
  
  # #############################################################################
  # # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # # Store the loss in loss and the gradient in dW. If you are not careful     #
  # # here, it is easy to run into numeric instability. Don't forget the        #
  # # regularization!                                                           #
  # #############################################################################
  
  score = W.dot(X)
  # On rajoute une constant pr ls overflow
  score += - np.max(score , axis=0)
  exp_score = np.exp(score) # matric exponientiel score
  sum_exp_score_col = np.sum(exp_score , axis = 0) # sum des expo score pr chaque column

  loss = np.log(sum_exp_score_col)
  loss = loss - score[y,np.arange(num_train)]
  loss = np.sum(loss) / float(num_train) + 0.5 * reg * np.sum(W*W)
  
  Grad = exp_score / sum_exp_score_col
  Grad[y,np.arange(num_train)] += -1.0
  dW = Grad.dot(X.T) / float(num_train) + reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
