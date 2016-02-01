import numpy as np
from random import shuffle
import sys



def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    indicator =  (scores-correct_class_score+1)>0
    for j in xrange(num_classes):
      if j == y[i]:
        dW[j,:] += -np.sum(np.delete(indicator,j))*X[:,i].T
        continue
      dW[j,:] += indicator[j]*X[:,i].T
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  
  # Add regularization to the loss and the gradient
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss,dW
  
def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  num_train = X.shape[1]
  dW = np.zeros(W.shape) # initialize the gradient as zero  
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  Loss = W.dot(X) - (W.dot(X))[y,np.arange(num_train)]+1
  Bool = Loss > 0
  # On sum sur les colonnes et on enlceve la valeur que l'on a compteur en trop
  Loss = np.sum(Loss * Bool , axis = 0) - 1.0
  Regularization = 0.5 * reg * np.sum(W*W)
  loss = np.sum(Loss) / float(num_train) +Regularization
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  Bool = Bool*np.ones(Loss.shape)
  Bool[[y,np.arange(num_train)]] = -(np.sum(Bool,axis=0)-1)
  dW = Bool.dot(X.T) / float(num_train)
  dW += reg * W
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
