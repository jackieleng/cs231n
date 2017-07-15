import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    delta = 1
    for i in xrange(num_train):
        loss_count = 0
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + delta
            if margin > 0:
                loss_count += 1
                dW[:, j] += X[i]
                loss += margin
        dW[:, y[i]] += -loss_count * X[i]

    # Loss rewritten in terms of W only (not entirely sure if correct)
    # for i in xrange(num_train):
    #     loss_count = 0
    #     for j in xrange(num_classes):
    #         margin = W[:, j].T.dot(X[i]) - W[:, y[i]].T.dot(X[i]) + delta
    #         if j != y[i] and margin > 0:
    #             loss += margin
    #             loss_count += 1
    #             dW[:, j] += X[i]
    #     dW[:, y[i]] += -loss_count * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################

    # compute the loss and the gradient
    num_train = X.shape[0]  # = N
    delta = 1.0
    scores = X.dot(W)  # NxC matrix of scores for each sample
    # Note: you need two arrays to index a matrix
    correct_class_scores = scores[np.arange(num_train), y]
    margins = np.maximum(
        0, scores - correct_class_scores[:, np.newaxis] + delta)
    margins[np.arange(num_train), y] = 0
    loss = np.sum(margins)  # total loss of all samples

    # Average loss over the number of samples
    loss /= num_train

    # Regularization
    loss += 0.5 * reg * np.sum(W * W)

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

    # Compute gradient
    loss_counts = np.zeros(margins.shape)  # NxC
    loss_counts[margins > 0] = 1
    # Nx1, loss counts for correct class
    total_loss_counts = np.sum(loss_counts, axis=1)
    loss_counts[np.arange(num_train), y] = -total_loss_counts
    dW = X.T.dot(loss_counts)  # (NxD).T * NxC = DxC

    # Average out weights
    dW /= num_train

    # Regularize the weights
    dW += reg * W

    # Half vectorized version:
    # for i in xrange(num_train):
    #     dW_i = dW_i_vectorized(X[i], y[i], W)
    #     dW += dW_i
    # dW /= num_train
    # dW += reg * W

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW


def dW_i_vectorized(x, y, W):
    """Only vectorized for one sample."""
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    delta = 1
    scores = x.dot(W)

    loss_counts = (scores - scores[y] + delta) > 0  # 1xC
    total_loss_count = np.sum(loss_counts)
    dW = x[:, np.newaxis].dot(loss_counts[np.newaxis, :])
    dW[:, y] = -total_loss_count * x
    return dW
