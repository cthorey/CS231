import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class FirstConvNet(object):
    """
    A L-layer convolutional network with the following architecture:

    [conv-relu-pool2x2]xL - [affine - relu]xM - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=[16, 32], filter_size=3,
                 hidden_dims=[100, 100], num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32, use_batchnorm=False):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: List of size  Nbconv+1 with the number of filters
        to use in each convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dims: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """

        self.use_batchnorm = use_batchnorm
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.bn_params = {}

        self.filter_size = filter_size
        self.L = len(num_filters)  # Number of weights
        self.M = len(hidden_dims)  # Number of conv/relu/pool blocks

        # Size of the input
        Cinput, Hinput, Winput = input_dim
        stride_conv = 1  # stride

        # Initialize the weight for the conv layers
        F = [Cinput] + num_filters
        for i in xrange(self.L):
            idx = i + 1
            W = weight_scale * \
                np.random.randn(
                    F[i + 1], F[i], self.filter_size, self.filter_size)
            b = np.zeros(F[i + 1])
            self.params.update({'W' + str(idx): W,
                                'b' + str(idx): b})
            if self.use_batchnorm:
                bn_param = {'mode': 'train',
                            'running_mean': np.zeros(F[i + 1]),
                            'running_var': np.zeros(F[i + 1])}
                gamma = np.ones(F[i + 1])
                beta = np.zeros(F[i + 1])
                self.bn_params.update({
                    'bn_param' + str(idx): bn_param})
                self.params.update({
                    'gamma' + str(idx): gamma,
                    'beta' + str(idx): beta})

        # Initialize the weights for the affine-relu layers
        # Size of the last activation
        Hconv, Wconv = self.Size_Conv(
            stride_conv, self.filter_size, Hinput, Winput, self.L)
        dims = [Hconv * Wconv * F[-1]] + hidden_dims
        for i in xrange(self.M):
            idx = self.L + i + 1
            W = weight_scale * \
                np.random.randn(dims[i], dims[i + 1])
            b = np.zeros(dims[i + 1])
            self.params.update({'W' + str(idx): W,
                                'b' + str(idx): b})
            if self.use_batchnorm:
                bn_param = {'mode': 'train',
                            'running_mean': np.zeros(dims[i + 1]),
                            'running_var': np.zeros(dims[i + 1])}
                gamma = np.ones(dims[i + 1])
                beta = np.zeros(dims[i + 1])
                self.bn_params.update({
                    'bn_param' + str(idx): bn_param})
                self.params.update({
                    'gamma' + str(idx): gamma,
                    'beta' + str(idx): beta})

        # Scoring layer
        W = weight_scale * np.random.randn(dims[-1], num_classes)
        b = np.zeros(num_classes)
        self.params.update({'W' + str(self.L + self.M + 1): W,
                            'b' + str(self.L + self.M + 1): b})

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def Size_Conv(self, stride_conv, filter_size, H, W, Nbconv):
        P = (filter_size - 1) / 2  # padd
        Hc = (H + 2 * P - filter_size) / stride_conv + 1
        Wc = (W + 2 * P - filter_size) / stride_conv + 1
        width_pool = 2
        height_pool = 2
        stride_pool = 2
        Hp = (Hc - height_pool) / stride_pool + 1
        Wp = (Wc - width_pool) / stride_pool + 1
        if Nbconv == 1:
            return Hp, Wp
        else:
            H = Hp
            W = Wp
            return self.Size_Conv(stride_conv, filter_size, H, W, Nbconv - 1)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        N = X.shape[0]

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = self.filter_size
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        if self.use_batchnorm:
            for key, bn_param in self.bn_params.iteritems():
                bn_param[mode] = mode

        scores = None
        #######################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #######################################################################

        blocks = {}
        blocks['h0'] = X
        # Forward into the conv blocks
        for i in xrange(self.L):
            idx = i + 1
            w = self.params['W' + str(idx)]
            b = self.params['b' + str(idx)]
            h = blocks['h' + str(idx - 1)]
            if self.use_batchnorm:
                beta = self.params['beta' + str(idx)]
                gamma = self.params['gamma' + str(idx)]
                bn_param = self.bn_params['bn_param' + str(idx)]
                h, cache_h = conv_norm_relu_pool_forward(
                    h, w, b, conv_param, pool_param, gamma, beta, bn_param)
            else:
                h, cache_h = conv_relu_pool_forward(
                    h, w, b, conv_param, pool_param)
            blocks['h' + str(idx)] = h
            blocks['cache_h' + str(idx)] = cache_h

        # Forward into the linear blocks
        for i in xrange(self.M):
            idx = self.L + i + 1
            h = blocks['h' + str(idx - 1)]
            if i == 0:
                h = h.reshape(N, np.product(h.shape[1:]))
            w = self.params['W' + str(idx)]
            b = self.params['b' + str(idx)]
            if self.use_batchnorm:
                beta = self.params['beta' + str(idx)]
                gamma = self.params['gamma' + str(idx)]
                bn_param = self.bn_params['bn_param' + str(idx)]
                h, cache_h = affine_norm_relu_forward(h, w, b, gamma,
                                                      beta, bn_param)
            else:
                h, cache_h = affine_relu_forward(h, w, b)
            blocks['h' + str(idx)] = h
            blocks['cache_h' + str(idx)] = cache_h

        # Fnally Forward into the score
        idx = self.L + self.M + 1
        w = self.params['W' + str(idx)]
        b = self.params['b' + str(idx)]
        h = blocks['h' + str(idx - 1)]
        h, cache_h = affine_forward(h, w, b)
        blocks['h' + str(idx)] = h
        blocks['cache_h' + str(idx)] = cache_h

        scores = blocks['h' + str(idx)]

        if y is None:
            return scores

        loss, grads = 0, {}
        #######################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #######################################################################

        # Computing of the loss
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0
        for w in [self.params[f] for f in self.params.keys() if f[0] == 'W']:
            reg_loss += 0.5 * self.reg * np.sum(w * w)

        loss = data_loss + reg_loss

        # Backward pass
        # print 'Backward pass'
        # Backprop into the scoring layer
        idx = self.L + self.M + 1
        dh = dscores
        h_cache = blocks['cache_h' + str(idx)]
        dh, dw, db = affine_backward(dh, h_cache)
        blocks['dh' + str(idx - 1)] = dh
        blocks['dW' + str(idx)] = dw
        blocks['db' + str(idx)] = db

        # Backprop into the linear blocks
        for i in range(self.M)[::-1]:
            idx = self.L + i + 1
            dh = blocks['dh' + str(idx)]
            h_cache = blocks['cache_h' + str(idx)]
            if self.use_batchnorm:
                dh, dw, db, dgamma, dbeta = affine_norm_relu_backward(
                    dh, h_cache)
                blocks['dbeta' + str(idx)] = dbeta
                blocks['dgamma' + str(idx)] = dgamma
            else:
                dh, dw, db = affine_relu_backward(dh, h_cache)
            blocks['dh' + str(idx - 1)] = dh
            blocks['dW' + str(idx)] = dw
            blocks['db' + str(idx)] = db

        # Backprop into the conv blocks
        for i in range(self.L)[::-1]:
            idx = i + 1
            dh = blocks['dh' + str(idx)]
            h_cache = blocks['cache_h' + str(idx)]
            if i == max(range(self.L)[::-1]):
                dh = dh.reshape(*blocks['h' + str(idx)].shape)
            if self.use_batchnorm:
                dh, dw, db, dgamma, dbeta = conv_norm_relu_pool_backward(
                    dh, h_cache)
                blocks['dbeta' + str(idx)] = dbeta
                blocks['dgamma' + str(idx)] = dgamma
            else:
                dh, dw, db = conv_relu_pool_backward(dh, h_cache)
            blocks['dh' + str(idx - 1)] = dh
            blocks['dW' + str(idx)] = dw
            blocks['db' + str(idx)] = db

        # w gradients where we add the regulariation term
        list_dw = {key[1:]: val + self.reg * self.params[key[1:]]
                   for key, val in blocks.iteritems() if key[:2] == 'dW'}
        # Paramerters b
        list_db = {key[1:]: val for key, val in blocks.iteritems() if key[:2] ==
                   'db'}
        # Parameters gamma
        list_dgamma = {key[1:]: val for key, val in blocks.iteritems() if key[
            :6] == 'dgamma'}
        # Paramters beta
        list_dbeta = {key[1:]: val for key, val in blocks.iteritems() if key[
            :5] == 'dbeta'}

        grads = {}
        grads.update(list_dw)
        grads.update(list_db)
        grads.update(list_dgamma)
        grads.update(list_dbeta)

        return loss, grads


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32, use_batchnorm=False):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.use_batchnorm = use_batchnorm
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.bn_params = {}

        #######################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        #######################################################################

        # Size of the input
        C, H, W = input_dim

        # Conv layer
        # The parameters of the conv is of size (F,C,HH,WW) with
        # F give the nb of filters, C,HH,WW characterize the size of
        # each filter
        # Input size : (N,C,H,W)
        # Output size : (N,F,Hc,Wc)
        F = num_filters
        filter_height = filter_size
        filter_width = filter_size
        stride_conv = 1  # stride
        P = (filter_size - 1) / 2  # padd
        Hc = (H + 2 * P - filter_height) / stride_conv + 1
        Wc = (W + 2 * P - filter_width) / stride_conv + 1

        W1 = weight_scale * np.random.randn(F, C, filter_height, filter_width)
        b1 = np.zeros(F)

        # Pool layer : 2*2
        # The pool layer has no parameters but is important in the
        # count of dimension.
        # Input : (N,F,Hc,Wc)
        # Ouput : (N,F,Hp,Wp)

        width_pool = 2
        height_pool = 2
        stride_pool = 2
        Hp = (Hc - height_pool) / stride_pool + 1
        Wp = (Wc - width_pool) / stride_pool + 1

        # Hidden Affine layer
        # Size of the parameter (F*Hp*Wp,H1)
        # Input: (N,F*Hp*Wp)
        # Output: (N,Hh)

        Hh = hidden_dim
        W2 = weight_scale * np.random.randn(F * Hp * Wp, Hh)
        b2 = np.zeros(Hh)

        # Output affine layer
        # Size of the parameter (Hh,Hc)
        # Input: (N,Hh)
        # Output: (N,Hc)

        Hc = num_classes
        W3 = weight_scale * np.random.randn(Hh, Hc)
        b3 = np.zeros(Hc)

        self.params.update({'W1': W1,
                            'W2': W2,
                            'W3': W3,
                            'b1': b1,
                            'b2': b2,
                            'b3': b3})

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.

        if self.use_batchnorm:
            print 'We use batchnorm here'
            bn_param1 = {'mode': 'train',
                         'running_mean': np.zeros(F),
                         'running_var': np.zeros(F)}
            gamma1 = np.ones(F)
            beta1 = np.zeros(F)

            bn_param2 = {'mode': 'train',
                         'running_mean': np.zeros(Hh),
                         'running_var': np.zeros(Hh)}
            gamma2 = np.ones(Hh)
            beta2 = np.zeros(Hh)

            self.bn_params.update({'bn_param1': bn_param1,
                                   'bn_param2': bn_param2})

            self.params.update({'beta1': beta1,
                                'beta2': beta2,
                                'gamma1': gamma1,
                                'gamma2': gamma2})

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        if self.use_batchnorm:
            for key, bn_param in self.bn_params.iteritems():
                bn_param[mode] = mode

        N = X.shape[0]

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        if self.use_batchnorm:
            bn_param1, gamma1, beta1 = self.bn_params[
                'bn_param1'], self.params['gamma1'], self.params['beta1']
            bn_param2, gamma2, beta2 = self.bn_params[
                'bn_param2'], self.params['gamma2'], self.params['beta2']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        #######################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #######################################################################

        # Forward into the conv layer
        x = X
        w = W1
        b = b1
        if self.use_batchnorm:
            beta = beta1
            gamma = gamma1
            bn_param = bn_param1
            conv_layer, cache_conv_layer = conv_norm_relu_pool_forward(
                x, w, b, conv_param, pool_param, gamma, beta, bn_param)
        else:
            conv_layer, cache_conv_layer = conv_relu_pool_forward(
                x, w, b, conv_param, pool_param)

        N, F, Hp, Wp = conv_layer.shape  # output shape

        # Forward into the hidden layer
        x = conv_layer.reshape((N, F * Hp * Wp))
        w = W2
        b = b2
        if self.use_batchnorm:
            gamma = gamma2
            beta = beta2
            bn_param = bn_param2
            hidden_layer, cache_hidden_layer = affine_norm_relu_forward(
                x, w, b, gamma, beta, bn_param)
        else:
            hidden_layer, cache_hidden_layer = affine_relu_forward(x, w, b)
        N, Hh = hidden_layer.shape

        # Forward into the linear output layer
        x = hidden_layer
        w = W3
        b = b3
        scores, cache_scores = affine_forward(x, w, b)

        if y is None:
            return scores

        loss, grads = 0, {}
        #######################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #######################################################################

        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.5 * self.reg * np.sum(W1**2)
        reg_loss += 0.5 * self.reg * np.sum(W2**2)
        reg_loss += 0.5 * self.reg * np.sum(W3**2)
        loss = data_loss + reg_loss

        # Backpropagation
        grads = {}
        # Backprop into output layer
        dx3, dW3, db3 = affine_backward(dscores, cache_scores)
        dW3 += self.reg * W3

        # Backprop into first layer
        if self.use_batchnorm:
            dx2, dW2, db2, dgamma2, dbeta2 = affine_norm_relu_backward(
                dx3, cache_hidden_layer)
        else:
            dx2, dW2, db2 = affine_relu_backward(dx3, cache_hidden_layer)

        dW2 += self.reg * W2

        # Backprop into the conv layer
        dx2 = dx2.reshape(N, F, Hp, Wp)
        if self.use_batchnorm:
            dx, dW1, db1, dgamma1, dbeta1 = conv_norm_relu_pool_backward(
                dx2, cache_conv_layer)
        else:
            dx, dW1, db1 = conv_relu_pool_backward(dx2, cache_conv_layer)

        dW1 += self.reg * W1

        grads.update({'W1': dW1,
                      'b1': db1,
                      'W2': dW2,
                      'b2': db2,
                      'W3': dW3,
                      'b3': db3})

        if self.use_batchnorm:
            grads.update({'beta1': dbeta1,
                          'beta2': dbeta2,
                          'gamma1': dgamma1,
                          'gamma2': dgamma2})

        #######################################################################
        #                             END OF YOUR CODE                             #
        #######################################################################

        return loss, grads


pass
