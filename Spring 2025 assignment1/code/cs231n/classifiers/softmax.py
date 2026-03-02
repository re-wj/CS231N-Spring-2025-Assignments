from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)    #prevent large scores overflow and will be canceled out when normalizing
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss

        ds = p - np.eye(num_classes)[y[i]]
        dW += np.dot(X[i][:, None], ds[None, :])


    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    #s = xW
    #s_j = \sum_{i=1}^d x_i W_{i,j}
    #ds_{k(k \neq j)}/dW_{i,j} = 0
    #ds_j/dW_{i,j} = x_i
    #dL/dW_{i,j} = \sum_{k=1}^n ds_k/dW_{i,j} dL/ds_k = x_i dL/ds_j
    #dL/dW = x^T dL/ds
    
    #p_j = e^{s_j} / \sum_{l=1}^n e^{s_l}
    #dp_{k(k \neq j)}/ds_j = -e^{s_j} e^{s_{k(k \neq j)}} / (\sum_{l=1}^n e^{s_l})^2 = -p_j p_{k(k \neq j)}
    #dp_j/ds_j = (e^{s_j} \sum_{l=1}^n e^{s_l} - e^{s_j} e^{s_{k(k \neq j)}}) / (\sum_{l=1}^n e^{s_l})^2 = -p_j p_{k(k \neq j)} = p_j (1-p_j)
    #dL/ds_j = \sum_{k=1}^n dp_k/ds_j dL/dp_k = p_j (dL/dp_j - \sum_{k=1}^n p_k dL/dp_k)
    #dL/ds = p * dL/dp - p (dL/dp p^T) = (diag(p) - p^T p) dL/dp
    
    #L = 1/N \sum_{l=1}^N -log p_{y_l} + lambda ||W||^2
    #d(-log p_{y_l})/dp_{k(k \neq y_l)} = 0
    #d(-log p_{y_l})/dp_{y_l} = -1/p_{y_l}
    #d(-log p_{y_l})/dp = -1_{y_l} / p

    #p can be cancled out when calculating d(-log p_{y_l})/ds directly
    #d(-log p_{y_l})/ds = p * (-1_{y_l} / p) - p ((-1_{y_l} / p) p^T)  = -1_{y_l} + p

    #d(lambda ||W||^2)/dW = 2lambda W

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
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################

    #X: (N, D)
    #W: (D, C)
    #y: (N, )

    s = np.dot(X, W)        #s: (N, C)
    s = s - np.max(s, axis=-1)[:, None]

    p = np.exp(s)           #p: (N, C)
    p = p / np.sum(p, axis=-1)[:, None]

    loss = -np.log(p[np.arange(X.shape[0]), y])
    loss = np.sum(loss) / X.shape[0] + reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################

    ds = p - np.eye(W.shape[1])[y]
    dW = np.dot(X.T, ds) / X.shape[0] + 2 * reg * W

    return loss, dW
