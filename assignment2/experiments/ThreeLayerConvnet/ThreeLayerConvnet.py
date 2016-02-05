import os
import json

DIR_CS231n = '/Users/cthorey/Documents/MLearning/CS231/assignment2/'
import sys
sys.path.append(DIR_CS231n)
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.cnn import ThreeLayerConvNet
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver

if __name__ == "__main__":
    conf_file = sys.argv[1]

    with open(conf_file, 'r') as f:
        conf = json.load(f)

    # Parameter extraction
    # Model instance

    input_dim = tuple(conf.get('input_dim', (3, 32, 32)))
    num_filters = conf.get('num_filters', 32)
    filter_size = conf.get('filter_size', 7)
    hidden_dim = conf.get('hidden_dim', 100)
    num_classes = conf.get('num_classes', 9)
    weight_scale = conf.get('weight_scale', 1e-3)
    reg = conf.get('reg', 0.0)
    dtype = conf.get('dtype', np.float32)
    use_batchnorm = conf.get('use_batchnorm', True)

    # Solver instance
    update_rule = conf.get('update_rule', 'adam')
    optim_config = conf.get('optim_config', {'learning_rate': 1e-3})
    lr_decay = conf.get('lr_decay', 1.0)
    batch_size = conf.get('batch_size', 100)
    num_epochs = conf.get('num_epochs', 10)
    print_every = conf.get('print_every', 10)
    verbose = conf.get('verbose', True)
    path = conf.get('path', '')

    if path == '':
        raise ValueError('You have to set a path where \
                         the model is suppose to run')

    # Load the (preprocessed) CIFAR10 data.
    data = get_CIFAR10_data(DIR_CS231n)
    for k, v in data.iteritems():
        print '%s: ' % k, v.shape

    print 'The parameters are: '
    for key, value in conf.iteritems():
        print key + ': ', value, ' \n'

    # Initialize the model instance
    model = ThreeLayerConvNet(input_dim=input_dim,
                              num_filters=num_filters,
                              filter_size=filter_size,
                              hidden_dim=hidden_dim,
                              num_classes=num_classes,
                              weight_scale=weight_scale,
                              reg=reg,
                              dtype=dtype,
                              use_batchnorm=use_batchnorm)

    # Run the training
    solver = Solver(model=model,
                    data=data,
                    path=path,
                    update_rule=update_rule,
                    optim_config=optim_config,
                    lr_decay=lr_decay,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    print_every=print_every,
                    verbose=verbose)

    solver.train()
