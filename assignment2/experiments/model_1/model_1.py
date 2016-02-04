import os
import json

DIR_CS231n = '/Users/thorey/Documents/MLearning/CS231/assignment2/'
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

    sys.exit()
    # Parameter extraction
    input_dim = conf['input_dim']
    num_filters = conf['num_filters']
    filter_size = conf['filter_size']
    hidden_dim = conf['hidden_dim']
    num_classes = conf['num_classes']
    weight_scale = conf['weight_scale']
    reg = conf['reg']
    dtype = conf['dtype']
    use_batchnorm = conf['use_batchnorm']
    update_rule = conf['update_rule'],
    optim_config = conf['optim_config'],
    lr_decay = conf['lr_decay'],
    batch_size = conf['batch_size'],
    num_epochs = conf['num_epochs'],
    print_every = conf['print_every'],
    verbose = conf['verbose']

    # Load the (preprocessed) CIFAR10 data.
    data = get_CIFAR10_data()
    for k, v in data.iteritems():
        print '%s: ' % k, v.shape

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
                    update_rule=update_rule,
                    optim_config=optim_config,
                    lr_decay=lr_decay,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    print_every=print_every,
                    verbose=verbose)

    solver.train()
