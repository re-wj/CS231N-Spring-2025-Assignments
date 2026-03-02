from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    out = x.reshape(x.shape[0], -1) @ w + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    dx = dout @ w.T
    dx = dx.reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T @ dout
    db = dout.sum(axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    dx = np.where(x>0, 1, 0) * dout
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    x = x - x.max(axis=-1)[:, None]

    p = np.exp(x)
    p = p / p.sum(axis=-1)[:, None]

    loss = -np.log(p[np.arange(x.shape[0]), y])
    loss = loss.sum(axis=0) / x.shape[0]

    dx = p - np.eye(x.shape[1])[y]
    dx = dx / x.shape[0]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################

        #mu_B = 1/B \sum_{i=1}^B x^{(i)}
        #sigma_B^2 = 1/B \sum_{i=1}^B (x^{(i)} - mu_B)
        sample_mean = np.mean(x, axis=0)    #mu_B: (D, )
        sample_var = np.var(x, axis=0)      #sigma_B^2: (D, )

        #x_caret = (x - 1^T mu_B) / sqrt{1^T sigma_B^2 + epsilon E}
        #o = x_caret * (1^T gamma) + 1^T beta
        x_caret = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = x_caret * gamma + beta
        
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        cache = (x, sample_mean, sample_var, x_caret, gamma, beta, eps)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################

        x_caret = (x - running_mean) / np.sqrt(running_var + eps)
        out = x_caret * gamma + beta

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    
    x, sample_mean, sample_var, x_caret, gamma, beta, eps = cache

    #-> mu_B -  -   |   -   beta -  |
    #|              v               v
    #x  -   -   ->  x_caret -   ->  o
    #|              ^               ^
    #-> sigma_B -   |   -   gamma - |

    #dL/dgamma_j = \sum_{k=1}^B \sum_{l=1}^D do^{(k)}_l/dgamma_j dL/do^{(k)}_l = \sum_{k=1}^B x_caret^{(k)}_j dL/do^{(k)}_j
    #dL/dgamma = \sum_{k=1}^B x_caret^{(k)} * dL/do^{(k)}
    dgamma = np.sum(x_caret * dout, axis=0)
    #dL/dbeta_j = \sum_{k=1}^B \sum_{l=1}^D do^{(k)}_l/dbeta_j dL/do^{(k)}_l = \sum_{k=1}^D dL/do^{(k)}_j
    #dL/dbeta = \sum_{k=1}^B dL/do^{(k)}
    dbeta = np.sum(dout, axis=0)
    #dL/dx_caret^{(i)}_j = \sum_{k=1}^B \sum_{l=1}^D do^{(k)}_l/dx_caret^{(i)}_j dL/do^{(k)}_l = gamma_j dL/do^{(i)}_j
    #dL/dx_caret = (1^T gamma) * dL/do
    dx_caret = gamma * dout

    #dsigma_B_j^2/dx^{(i)}_j = 2/B (x^{(i)}_j - mu_B_j)
    #dsigma_B_{l(l \neq j)}^2/dx^{(i)}_j = 0
    #dmu_B_j/dx^{(i)}_j = 1/B
    #dmu_B_{k(k \neq j)}/dx^{(i)}_j = 0
    #dx_caret^{(i)}_j/dx^{(i)}_j = 1/sqrt{sigma_B_j^2 + epsilon}
    #dx_caret^{(i)}_{l(l \neq j)}/dx^{(i)}_j = 0
    #dx_caret^{(k(k \neq i))}_{j}/dx^{(i)}_j = 0

    #dx_caret^{(i)}_j/dmu_B_j = -1/sqrt{sigma_B_j^2 + epsilon}
    #dx_caret^{(i)}_{l(l \neq j)}/dmu_B_j = 0
    #dx_caret^{(i)}_j/dsigma_B_j = -1/2 (x^{(i)}_j - mu_B_j) / (sqrt{sigma_B_j^2 + epsilon})^3
    #dx_caret^{(i)}_{l(l \neq j)}/dsigma_B_j = 0

    #dL/dmu_B_j = \sum_{k=1}^B \sum_{l=1}^D dx_caret^{(k)}_l/dmu_B_j dL/dx_caret^{(k)}_l = \sum_{k=1}^B -1/sqrt{sigma_B_j^2 + epsilon} dL/dx_caret^{(k)}_j
    #dL/dsigma_B_j = \sum_{k=1}^B \sum_{l=1}^D dx_caret^{(k)}_l/dsigma_B_j dL/dx_caret^{(k)}_l = \sum_{k=1}^B -1/2 (x^{(k)}_j - mu_B_j) / (sqrt{sigma_B_j^2 + epsilon})^3 dL/dx_caret^{(k)}_j
    dmu = -dx_caret / np.sqrt(sample_var + eps)
    dvar_2 = (x - sample_mean) * dmu / (sample_var + eps)
    dmu = np.sum(dmu, axis=0)
    dvar_2 = np.sum(dvar_2, axis=0)

    #dL/dx^{(i)}_j = \sum_{l=1}^D dmu_B_l/dx^{(i)}_j dL/dmu_B_l + \sum_{l=1}^D dsigma_B_l/dx^{(i)}_j dL/dsigma_B_l + \sum_{k=1}^B \sum_{l=1}^D dx_caret^{(k)}_l/dx^{(i)}_j dL/dx_caret^{(k)}_l
    #              = 1/B dL/dmu_B_j + 2/B (x^{(i)}_j - mu_B_j) dL/dsigma_B_j + 1/sqrt{sigma_B_j^2 + epsilon} dL/dx_caret^{(i)}_j
    #dL/dx = 1/B 1^T dL/dmu_B + 2/B (x - 1^T mu_B) * 1^T dL/dsigma_B + 1 / (sqrt{1^T sigma_B^2 + epsilon E}) dL/dx_caret
    dx = dmu / x.shape[0] + (x - sample_mean) * dvar_2 / x.shape[0] + dx_caret / np.sqrt(sample_var + eps)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    x, sample_mean, sample_var, x_caret, gamma, beta, eps = cache

    dgamma = np.sum(x_caret * dout, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx_caret = gamma * dout

    #dL/dx^{(i)}_j = 1/B dL/dmu_B_j + 2/B (x^{(i)}_j - mu_B_j) dL/dsigma_B_j + 1/sqrt{sigma_B_j^2 + epsilon} dL/dx_caret^{(i)}_j
    #              = -1/B 1/sqrt{sigma_B_j^2 + epsilon} \sum_{k=1}^B dL/dx_caret^{(k)}_j - 1/B 1/sqrt{sigma_B_j^2 + epsilon} \sum_{k=1}^B (x^{(i)}_j - mu_B_j)(x^{(k)}_j - mu_B_j) / (sigma_B_j^2 + epsilon) dL/dx_caret^{(k)}_j + 1/sqrt{sigma_B_j^2 + epsilon} dL/dx_caret^{(i)}_j
    #              = 1/sqrt{sigma_B_j^2 + epsilon} (dL/dx_caret^{(i)}_j - 1/B \sum_{k=1}^B dL/dx_caret^{(k)}_j - 1/B \sum_{k=1}^B (x^{(i)}_j - mu_B_j)(x^{(k)}_j - mu_B_j) / (sigma_B_j^2 + epsilon) dL/dx_caret^{(k)}_j)
    #              = 1/sqrt{sigma_B_j^2 + epsilon} (dL/dx_caret^{(i)}_j - 1/B \sum_{k=1}^B dL/dx_caret^{(k)}_j - 1/B x_caret^{(i)}_j \sum_{k=1}^B x_caret^{(k)}_j dL/dx_caret^{(k)}_j)
    #dL/dx = 1/sqrt{1^T sigma_B^2 + epsilon E} (dL/dx_caret - 1^T (1/B \sum_{k=1}^B dL/dx_caret^{(k)}) - x_caret * (1^T (1/B \sum_{k=1}^B x_caret^{(k)} * dL/dx_caret^{(k)})))
    dx = (dx_caret - np.mean(dx_caret, axis=0) - x_caret * np.mean(x_caret * dx_caret, axis=0)) / np.sqrt(sample_var + eps)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and shift the normalized data    #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    
    #mu_L = 1/D \sum_{j=1}^D x_j
    #sigma_D^2 = 1/D \sum_{j=1}^N (x_j - mu_L)
    sample_mean = np.mean(x, axis=-1)   #mu_L: (N, )
    sample_var = np.var(x, axis=-1)     #sigma_L^2: (N, )
    
    #x_caret = (x - (mu_L)^T 1) / sqrt{(sigma_L^2)^T 1 + epsilon E}
    #o = x_caret * (1^T gamma) + 1^T beta
    x_caret = (x - sample_mean[:, None]) / np.sqrt(sample_var[:, None] + eps)
    out = x_caret * gamma + beta

    cache = (x, sample_mean, sample_var, x_caret, gamma, beta, eps)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################

    x, sample_mean, sample_var, x_caret, gamma, beta, eps = cache

    #-> mu_L -  -   |   -   beta -  |
    #|              v               v
    #x  -   -   ->  x_caret -   ->  o
    #|              ^               ^
    #-> sigma_L -   |   -   gamma - |

    #dL/dgamma_j = \sum_{k=1}^B \sum_{l=1}^D do^{(k)}_l/dgamma_j dL/do^{(k)}_l = \sum_{k=1}^B x_caret^{(k)}_j dL/do^{(k)}_j
    #dL/dgamma = \sum_{k=1}^B x_caret^{(k)} * dL/do^{(k)}
    dgamma = np.sum(x_caret * dout, axis=0)
    #dL/dbeta_j = \sum_{k=1}^B \sum_{l=1}^D do^{(k)}_l/dbeta_j dL/do^{(k)}_l = \sum_{k=1}^D dL/do^{(k)}_j
    #dL/dbeta = \sum_{k=1}^B dL/do^{(k)}
    dbeta = np.sum(dout, axis=0)
    #dL/dx_caret^{(i)}_j = \sum_{k=1}^B \sum_{l=1}^D do^{(k)}_l/dx_caret^{(i)}_j dL/do^{(k)}_l = gamma_j dL/do^{(i)}_j
    #dL/dx_caret = (1^T gamma) * dL/do
    dx_caret = gamma * dout
    
    #dsigma_L^{(i)}^2/dx^{(i)}_j = 2/D (x^{(i)}_j - mu_L^{(i)})
    #dsigma_L^{(k(k \neq i))}^2/dx^{(i)}_j = 0
    #dmu_L^{(i)}/dx^{(i)}_j = 1/D
    #dmu_L_^{(k(k \neq i))}/dx^{(i)}_j = 0
    #dx_caret^{(i)}_j/dx^{(i)}_j = 1/sqrt{sigma_L^{(i)}^2 + epsilon}
    #dx_caret^{(k(k \neq i))}_{j}/dx^{(i)}_j = 0
    #dx_caret^{(i)}_{l(l \neq j)}/dx^{(i)}_j = 0

    #dx_caret^{(i)}_j/dmu_L^{(i)} = -1/sqrt{sigma_L^{(i)}^2 + epsilon}
    #dx_caret^{(k(k \neq i))}_j/dmu_L^{(i)} = 0
    #dx_caret^{(i)}_j/dsigma_L^{(i)} = -1/2 (x^{(i)}_j - mu_L^{(i)}) / (sqrt{sigma_L^{(i)}^2 + epsilon})^3
    #dx_caret^{(k(k \neq i))}_j/dsigma_L^{(i)} = 0

    #dL/dmu_L^{(i)} = \sum_{k=1}^B \sum_{l=1}^D dx_caret^{(k)}_l/dmu_B_j dL/dx_caret^{(k)}_l = \sum_{l=1}^D -1/sqrt{sigma_L^{(i)}^2 + epsilon} dL/dx_caret^{(i)}_l
    #dL/dsigma_L^{(i)} = \sum_{k=1}^B \sum_{l=1}^D dx_caret^{(k)}_l/dsigma_B_j dL/dx_caret^{(k)}_l = \sum_{l=1}^D -1/2 (x^{(i)}_l - mu_L^{(i)}) / (sqrt{sigma_L^{(i)}^2 + epsilon})^3 dL/dx_caret^{(i)}_l

    #dL/dx^{(i)}_j = \sum_{k=1}^B dmu_L^{(k)}/dx^{(i)}_j dL/dmu_L^{(k)} + \sum_{k=1}^D dsigma_L^{(i)}/dx^{(i)}_j dL/dsigma_L^{(i)} + \sum_{k=1}^B \sum_{l=1}^D dx_caret^{(k)}_l/dx^{(i)}_j dL/dx_caret^{(k)}_l
    #              = 1/D dL/dmu_L^{(i)} + 2/D (x^{(i)}_j - mu_L^{(i)}) dL/dsigma_L^{(i)} + 1/sqrt{sigma_L^{(i)}^2 + epsilon} dL/dx_caret^{(i)}_j
    #              = -1/D 1/sqrt{sigma_L^{(i)}^2 + epsilon} \sum_{l=1}^D dL/dx_caret^{(i)}_l - 1/D 1/sqrt{sigma_L^{(i)}^2 + epsilon} \sum_{l=1}^D (x^{(i)}_j - mu_L^{(i)})(x^{(i)}_l - mu_L^{(i)}) / (sigma_L^{(i)}^2 + epsilon) dL/dx_caret^{(i)}_l + 1/sqrt{sigma_L^{(i)}^2 + epsilon} dL/dx_caret^{(i)}_j
    #              = 1/sqrt{sigma_L^{(i)}^2 + epsilon} (dL/dx_caret^{(i)}_j - 1/D \sum_{l=1}^D dL/dx_caret^{(i)}_l - 1/D \sum_{l=1}^D (x^{(i)}_j - mu_L^{(i)})(x^{(i)}_j - mu_L^{(i)}) / (sigma_L^{(i)}^2 + epsilon) dL/dx_caret^{(i)}_l)
    #              = 1/sqrt{sigma_L^{(i)}^2 + epsilon} (dL/dx_caret^{(i)}_j - 1/D \sum_{l=1}^D dL/dx_caret^{(i)}_l - 1/D x_caret^{(i)}_j \sum_{l=1}^D x_caret^{(i)}_l dL/dx_caret^{(i)}_l)
    #dL/dx = 1/sqrt{(sigma_L^2)^T 1 + epsilon E} (dL/dx_caret - (1/D \sum_{l=1}^D dL/dx_caret_l)^T 1 - x_caret * ((1/D \sum_{l=1}^D x_caret_l * dL/dx_caret_l)^T 1))
    dx = (dx_caret - np.mean(dx_caret, axis=-1)[:, None] - x_caret * np.mean(x_caret * dx_caret, axis=-1)[:, None]) / np.sqrt(sample_var[:, None] + eps)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################

        mask = np.random.binomial(1, p, size=x.shape)
        out = x * mask / p

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################

        out = x

        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################

        dx = dout * mask / dropout_param["p"]

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    pad = conv_param["pad"]
    s = conv_param["stride"]
    H_, W_ = (1+(H+2*pad-HH)//s, 1+(W+2*pad-WW)//s)

    x_padded = np.pad(x,((0, 0), (0, 0), (pad, pad), (pad, pad)) , mode="constant", constant_values=0)  #x_padded: (N, C, H+2*pad, W+2*pad)
    
    #o_{f, i, j} = \sum_{k=1}^HH \sum_{l=1}^WW \sum_{c=1}^C x_{s*(i-1)+k, s*(j-1)+l, c} w_{f, k, l, c} + b_f
    out = np.empty((N, F, H_, W_))
    w_flattened = w.reshape(F, -1)                          #w_flattened: (F, C*HH*WW)
    for i in range(H_):
        for j in range(W_):
            x_window = x_padded[:,:,i*s:i*s+HH,j*s:j*s+WW]  #x_window: (N, C, HH, WW)
            out[:, :, i, j] = x_window.reshape(N, -1) @ w_flattened.T
    out+= b.reshape(1, -1, 1, 1)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    pad = conv_param["pad"]
    s = conv_param["stride"]
    H_, W_ = dout.shape[2:]

    #(1 \leq i-s*(k-1) \leq HH \land 1 \leq j-s*(l-1) \leq WW) do_{f, k, l}/dx_{i, j, c} = w_{f, i-s*(k-1), j-s*(l-1), c}
    #(else) do_{f, k, l}/dx_{i, j, c} = 0
    #dL/dx_{i, j, c} = \sum_{f=1}^F \sum_{k=1}^H' \sum_{l=1}^W' do_{f, k, l}/dx_{i, j, c} dL/do_{f, k, l}
    #                = \sum_{f=1}^F \sum_{1 \leq i-s*(k-1) \leq HH} \sum_{1 \leq j-s*(l-1) \leq WW} w_{f, i-s*(k-1), j-s*(l-1), c} dL/do_{f, k, l}
    #                = \sum_{f=1}^F \sum_{1 \leq k \leq HH \land s|(i-k)} \sum_{1 \leq l \leq WW \land s|(j-l)} w_{f, k, l, c} dL/do_{f, 1+(i-k)/s, 1+(j-l)/s}
    #too complicated

    #(1 \leq k-s*(i-1) \leq HH \land 1 \leq l-s*(j-1) \leq WW) do_{f, i, j}/dx_{k, l, c} = w_{f, k-s*(i-1), l-s*(j-1),c}
    #(else) do_{f, i, j}/dx_{k, l, c} = 0
    #(1 \leq k-s*(i-1) \leq HH \land 1 \leq l-s*(j-1) \leq WW) \Delta_{dL/do_{f, i, j}} dL/dx_{k, l, c} = do_{f, i, j}/dx_{k, l, c} dL/do_{f, i, j} = \sum_{f=1}^F w_{f, k-s*(i-1), l-s*(j-1), c} dL/do_{f, i, j}
    #(1 \leq k \leq HH \land 1 \leq l \leq WW) \Delta_{dL/do_{f, i, j}} dL/dx_{k+s*(i-1), l+s*(j-1), c} = \sum_{f=1}^F w_{f, k, l, c} dL/do_{f, i, j}
    #(else) \Delta_{dL/do_{f, i, j}} dL/dx_{k+s*(i-1), l+s*(j-1), c} = 0
    #dL/dx_{i, j, c} = \sum_{f=1}^F \sum_{k=1}^H' \sum_{l=1}^W' do_{f, k, l}/dx_{i, j, c} dL/do_{f, k, l}
    #                = \sum_{f=1}^F \sum_{1 \leq k \leq HH \land s|(i-k)} \sum_{1 \leq l \leq WW \land s|(j-l)} \Delta_{dL/do_{f, k, l}} dL/dx_{i, j, c}
    dx_padded = np.zeros((N, C, H+2*pad, W+2*pad))
    w_flattened = w.reshape(F, -1)              #w_flattened: (F, C*HH*WW)
    for i in range(H_):
        for j in range(W_):
            dout_window = dout[:, :, i, j]      #dout_window: (N, F)
            dx_padded[:,:,i*s:i*s+HH,j*s:j*s+WW] += np.reshape(dout_window @ w_flattened, (N, C, HH, WW))
    dx = dx_padded[:, :, pad:H+pad, pad:W+pad]
    
    #do_{f, k, l}/dw_{f, i, j, c} = x_{s*(k-1)+i, s*(l-1)+j, c}
    #do_{f'(f' \neq f), k, l}/dw_{f, i, j, c} = 0
    #dL/dw_{f, i, j, c} = \sum_{f'=1}^F \sum_{k=1}^H' \sum_{l=1}^W' do_{f', k, l}/dw_{f, i, j, c} dL/do_{f', k, l}
    #                   = \sum_{k=1}^H' \sum_{l=1}^W' x_{s*(k-1)+i, s*(l-1)+j, c} dL/do_{f, k, l}

    #do_{f, i, j}/dw_{f, k, l, c} = x_{s*(i-1)+k, s*(j-1)+l, c}
    #do_{f, i, j}/dw_{f'(f' \neq f), k, l, c} = 0
    #\Delta_{dL/do_{f, i, j}} dL/dw_{f, k, l, c} = do_{f, i, j}/dw_{f, k, l, c} dL/do_{f, i, j} = x_{s*(i-1)+k, s*(j-1)+l, c} dL/do_{f, i, j}
    #\Delta_{dL/do_{f, i, j}} dL/dw_{f'(f' \neq f), k, l, c} = 0
    #dL/dw_{f, i, j, c} = \sum_{f'=1}^F \sum_{k=1}^H' \sum_{l=1}^W' do_{f', k, l}/dw_{f, i, j, c} dL/do_{f', k, l}
    #                   = \sum_{k=1}^H' \sum_{l=1}^W' \Delta_{dL/do_{f, i, j}} dL/dw_{f, k, l, c}
    x_padded = np.pad(x,((0, 0), (0, 0), (pad, pad), (pad, pad)) , mode="constant", constant_values=0)
    dw = np.zeros_like(w)
    for i in range(H_):
        for j in range(W_):
            x_window = x_padded[:,:,i*s:i*s+HH,j*s:j*s+WW]  #x_window: (N, C, HH, WW)
            dout_window = dout[:, :, i, j]                  #dout_window: (N, F)
            dw += np.reshape(dout_window.T @ x_window.reshape(N, -1), (F, C, HH, WW))

    #dL/db_f = \sum_{k=1}^H' \sum_{l=1}^W' do_{f, k, l}/db_f dL/do_{f, k, l} = \sum_{k=1}^H' \sum_{l=1}^W' dL/do_{f, k, l}
    db = np.sum(dout, axis=(0, -1, -2))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    
    N, C, H, W = x.shape
    HH, WW = (pool_param["pool_height"], pool_param["pool_width"])
    s = pool_param["stride"]
    H_, W_ = (1+(H-HH)//s, 1+(W-WW)//s)
    
    #o_{i, j, c} = \max_{\substack{1 \leq k \leq HH \\ 1 \leq l \leq WW}} {x_{s*(i-1)+k, s*(j-1)+l, c}}
    out = np.empty((N, C, H_, W_))
    for i in range(H_):
        for j in range(W_):
            x_window = x[:,:,i*s:i*s+HH,j*s:j*s+WW]  #x_window: (N, C, HH, WW)
            out[:, :, i, j] = np.max(x_window, axis=(-1, -2))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################

    x, pool_param = cache
    N, C, H, W = x.shape
    HH, WW = (pool_param["pool_height"], pool_param["pool_width"])
    s = pool_param["stride"]
    H_, W_ = (1+(H-HH)//s, 1+(W-WW)//s)

    #M_{i, j, c} = card({x_{k, l, c} | 1 \leq k-s*(i-1) \leq HH \land 1 \leq l-s*(j-1) \leq WW \land x_{k, l, c} = \max_{\substack{1 \leq k' \leq HH \\ 1 \leq l' \leq WW}} {x_{s*(i-1)+k', s*(j-1)+l', c}}})
    #            = card({x_{k, l, c} | 1 \leq k-s*(i-1) \leq HH \land 1 \leq l-s*(j-1) \leq WW \land x_{k, l, c} = o_{i, j, c}})

    #(1 \leq i-s*(k-1) \leq HH \land 1 \leq j-s*(l-1) \leq WW \land x_{i, j, c} = o_{k, l, c}) do_{k, l, c}/dx_{i, j, c} = 1/M_{k, l, c}
    #(else) do_{k, l, c'}/dx_{i, j, c} = 0

    #(1 \leq k-s*(i-1) \leq HH \land 1 \leq l-s*(j-1) \leq WW \land x_{k, l, c} = o_{i, j, c}) do_{i, j, c}/dx_{k, l, c} = 1/M_{i, j, c}
    #(else) do_{i, j, c}/dx_{k, l, c'} = 0
    #(1 \leq k-s*(i-1) \leq HH \land 1 \leq l-s*(j-1) \leq WW \land x_{k, l, c} = o_{i, j, c}) \Delta_{dL/do_{i, j, c}} dL/dx_{k, l, c} = do_{i, j, c}/dx_{k, l, c} dL/do_{i, j, c} = 1/M_{i, j, c} dL/do_{i, j, c}
    #(1 \leq k \leq HH \land 1 \leq l \leq WW) \Delta_{dL/do_{f, i, j}} dL/dx_{k+s*(i-1), l+s*(j-1), c} = \sum_{f=1}^F w_{f, k, l, c} dL/do_{f, i, j}
    #(else) \Delta_{dL/do_{f, i, j}} dL/dx_{k+s*(i-1), l+s*(j-1), c} = 0
    #dL/dx_{i, j, c} = \sum_{k=1}^H' \sum_{l=1}^W' \sum_{c'=1}^C do_{k, l, c'}/dx_{i, j, c} dL/do_{k, l, c'}
    #                = \sum_{c'=1}^C \sum_{1 \leq k \leq HH \land s|(i-k)} \sum_{\substack{1 \leq l \leq WW \land s|(j-l) \\ x_{k, l, c} = o_{i, j, c}}} \Delta_{dL/do_{k, l, c}} dL/dx_{i, j, c}
    dx = np.zeros_like(x)
    for i in range(H_):
        for j in range(W_):
            x_window = x[:,:,i*s:i*s+HH,j*s:j*s+WW] #x_window: (N, C, HH, WW)
            max_position_mask = x_window == np.max(x_window, axis=(-1, -2))[:, :, None, None]
            max_position_mask = max_position_mask/max_position_mask.sum(axis=(-1, -2))[:, :, None, None]
            dout_window = dout[:, :, i, j]          #dout_window: (N, C)
            dx[:,:,i*s:i*s+HH,j*s:j*s+WW] += max_position_mask * dout_window[:, :, None, None]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################

    N, C, H, W = x.shape
    x = x.transpose((0, 2, 3, 1)).reshape(-1, C)
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose((0, 3, 1, 2))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    
    N, C, H, W = dout.shape
    dout = dout.transpose((0, 2, 3, 1)).reshape(-1, C)
    dx, dgamma, dbeta = batchnorm_backward_alt(dout, cache)
    dx = dx.reshape(N, H, W, C).transpose((0, 3, 1, 2))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer number of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    
    N, C, H, W = x.shape
    if C % G != 0:
        raise ValueError(f"Number of groups should be a divisor of channels: {G} is not a divisor of {C}")
    x = x.reshape((N, G, -1))           #x: (N, G, C/G * H * W)


    sample_mean = np.mean(x, axis=-1)   #mu_G: (N, G)
    sample_var = np.var(x, axis=-1)     #sigma_G^2: (N, G)
    
    x_caret = (x - sample_mean[:, :, None]) / np.sqrt(sample_var[:, :, None] + eps)

    x_caret = x_caret.reshape((N, C, H, W))
    out = x_caret * gamma + beta

    cache = (x, G, sample_mean, sample_var, x_caret, gamma, beta, eps)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    
    x, G, sample_mean, sample_var, x_caret, gamma, beta, eps = cache

    N, C, H, W = dout.shape

    dgamma = np.sum(x_caret * dout, axis=(0, -1, -2)).reshape(1, C, 1, 1)
    dbeta = np.sum(dout, axis=(0, -1, -2)).reshape(1, C, 1, 1)
    dx_caret = gamma * dout
    
    x_caret = x_caret.reshape((N, G, -1))       #x_caret: (N, G, C/G * H * W)
    dx_caret = dx_caret.reshape((N, G, -1))     #dx_caret: (N, G, C/G * H * W)
    dx = (dx_caret - np.mean(dx_caret, axis=-1)[:, :, None] - x_caret * np.mean(x_caret * dx_caret, axis=-1)[:, :, None]) / np.sqrt(sample_var[:, :, None] + eps)
    dx = dx.reshape((N, C, H, W))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
