import os
import json
import numpy as np

DIR_CS231n = '/Users/thorey/Documents/MLearning/CS231/assignment2/'

conf = {}

# Model instance
conf['input_dim'] = (3, 32, 32)
conf['num_filters'] = 32
conf['filter_size'] = 7
conf['hidden_dim'] = 100
conf['num_classes'] = 10
conf['weight_scale'] = 1e-3
conf['use_batchnorm'] = True

# Solver instance
conf['update_rule'] = 'adam'
conf['optim_config'] = {'learning_rate': 1e-3}
conf['lr_decay'] = 1.0
conf['batch_size'] = 100
conf['num_epochs'] = 10
conf['print_every'] = 10
conf['verbose'] = True

name_json = os.path.join(DIR_CS231n, 'experiments',
                         'ThreeLayerConvnet', 'model_0')
with open(name_json + '.json', 'w+') as f:
    json.dump(conf,
              f,
              sort_keys=True,
              indent=4,
              ensure_ascii=False)
