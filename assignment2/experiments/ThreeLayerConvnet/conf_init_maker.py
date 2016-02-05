import os
import sys
from sklearn.externals import joblib
import json
import numpy as np

DIR_CS231n = '/Users/cthorey/Documents/MLearning/CS231/assignment2/'

conf = {}

# Model instance
conf['input_dim'] = (3, 32, 32)
conf['num_filters'] = 64
conf['filter_size'] = 3
conf['hidden_dim'] = 500
conf['num_classes'] = 10
conf['weight_scale'] = 1e-3
conf['use_batchnorm'] = True

# Solver instance
conf['update_rule'] = 'adam'
conf['lr_decay'] = 0.99
conf['batch_size'] = 50
conf['num_epochs'] = 1
conf['print_every'] = 10
conf['verbose'] = True

# Helper function


def name_model(path):
    ''' Given a directory where you want to run a new model
    automatically select the name of the model by incrementing
    by 1 the largest previous model in the name'''

    existing_models = [f for f in os.listdir(
        path) if f.split('_')[0] == 'model']
    if len(existing_models) == 0:
        model = -1
    else:
        model = max([int(f.split('_')[1]) for f in existing_models])
    return os.path.join(path, 'model_' + str(model + 1))

name = os.listdir(DIR_CS231n)
dir_json = name_model(os.path.join(
    DIR_CS231n, 'experiments', 'ThreeLayerConvnet'))

conf['path'] = dir_json

try:
    'Initialize the model tree'
    os.mkdir(dir_json)
except:
    raise ValueError(
        'Cannot create the directory for the model %s' % (dir_json))

with open(os.path.join(dir_json, 'conf_init.json'), 'w+') as f:
    json.dump(conf,
              f,
              sort_keys=True,
              indent=4,
              ensure_ascii=False)
