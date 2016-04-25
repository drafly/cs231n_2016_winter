import numpy as np
import numpy.matlib
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  dWp = np.zeros((num_train, num_classes))
    
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores = scores - scores.max()
    scores = np.exp(scores)
    p = scores / scores.sum()
    groundTruth = np.zeros(p.shape)
    
    for j in xrange(num_classes):
      if j == y[i]:
        loss += -np.log(p[j])
        groundTruth[j] = 1
        
    # After for j finishes
    dWp[i,:] = groundTruth - p
    
  # After for finishes
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  dW = (-1) * X.T.dot(dWp)
  dW /= num_train
  dW += reg * W
    
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  Num_train = X.shape[0]
  Num_classes = W.shape[1]
  scores = X.dot(W)
  scores = scores.T
  scores = scores - scores.max(axis=0)
  scores = np.exp(scores)
  p = scores/scores.sum(axis=0)
    
  loss = np.sum(-np.log(p[y, range(Num_train)]))
  loss /= Num_train
  loss += 0.5*reg*np.sum(W**2)  

  groundTruth = np.zeros((Num_classes, Num_train))
  groundTruth[y, range(Num_train)] = 1
  dW = (-1) * (groundTruth - p).dot(X).T
  dW /= Num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

