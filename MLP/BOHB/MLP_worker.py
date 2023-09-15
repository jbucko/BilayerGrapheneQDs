import numpy as np
import time

import ConfigSpace as CS
from hpbandster.core.worker import Worker
import sklearn.neural_network as NN



class MLP_Worker(Worker):

    def __init__(self, *args, sleep_interval=0, x_train = None, y_train = None,x_validate = None, y_validate = None,  **kwargs):
        super().__init__(*args, **kwargs)

        self.sleep_interval = sleep_interval
        self.x_train = x_train
        self.y_train = y_train
        self.x_validate = x_validate
        self.y_validate = y_validate
    def compute(self, config, budget, **kwargs):
        """
        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """

        """
        classifier definition
        """
        clf = NN.MLPClassifier(verbose = 1,learning_rate = 'constant',hidden_layer_sizes = (800,800),max_iter = int(budget),\
                early_stopping = False, alpha = config['reg'], learning_rate_init =  config['lr'])

        print('l_rate, reg_strength:',config['lr'],config['reg'])
        batch_size = 500
        n_iter_no_change = 50
        loss_validate = []

        max_score = 0.
        k_best = 0
        batch_min = 0
        """
        perform budget of epochs (batches)
        """
        for k in range(int(budget)):
            if (k+1)*batch_size > self.x_train.shape[0]:
                batch_min = 0
            clf.partial_fit(self.x_train[batch_min:batch_min + batch_size,:],self.y_train[batch_min:batch_min + batch_size],np.unique(self.y_train))
            # print('fitted')
            score = clf.score(self.x_validate,self.y_validate)
            # print('validated')
            batch_min += batch_size
            loss_validate.append(score)
            if score > max_score:
                max_score = score
                k_best = k
            if k - k_best > n_iter_no_change:
                print('early stopping...')
                break

        print('validation scores for l_rate: {}, reg_strength: {} -> '.format(config['lr'],config['reg']),loss_validate)

        """define your loss"""
        res = 1 - loss_validate[-1]
        time.sleep(self.sleep_interval)

        return({
                    'loss': float(res),  # this is the a mandatory field to run hyperband
                    'info': res  # can be used for any user-defined information - also mandatory
                })
    
    """
    define your configuration space
    """
    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('lr', lower=1e-7, upper=1.0,log = True))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('reg', lower=1e-7, upper=1.0,log = True))
        return(config_space)

