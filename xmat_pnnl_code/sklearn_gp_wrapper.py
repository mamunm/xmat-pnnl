import numpy as np
from sklearn.metrics import (mean_absolute_error, 
        mean_squared_error, r2_score)
from importlib import import_module
import matplotlib.pyplot as plt

class SKGP():
    '''An object to oversee a gp run and store all the information pertinent
    to that run as a self contained source.'''

    def __init__(self,
                X=None,
                y=None,
                kernel_recipe=None,
                kernel=None,
                estimator_param=None,
                validation=None):

        self.X = X
        self.y = y
        self.kernel_recipe = kernel_recipe
        self.kernel = kernel
        self.estimator_param = estimator_param
        self.validation = validation

    def run_GP(self):

        est = import_module('sklearn.gaussian_process')
        estimator = getattr(est, 'GaussianProcessRegressor')

        if self.kernel is None:
            kernel = SKGP._cook_kernel(self.kernel_recipe)
        else:
            kernel = self.kernel

        if not self.estimator_param:
            estimator = estimator(kernel=kernel,
                                  n_restarts_optimizer=8,
                                  alpha=0)
        else:
            if 'kernel' not in self.estimator_param:
                 estimator = estimator(kernel=kernel,
                                       **self.estimator_param)
            else:
                 estimator = estimator(**self.estimator_param)
        
        self.model = estimator.fit(self.X, self.y)
        #Model Validation
        if self.validation == 'leave_one_out':
            val = getattr(import_module('sklearn.model_selection'), 
                    'LeaveOneOut')()
        else:
            val = getattr(import_module('sklearn.model_selection'), 
                    'KFold')(n_splits=int(self.validation.split('-')[0]))
            
        y_true_test = []
        y_pred_test = []
        y_pred_test_unc = []
        y_true_train = []
        y_pred_train = []
        y_pred_train_unc = []
        for tr_id, ts_id in val.split(self.y):
            XTR, XTS, YTR = self.X[tr_id], self.X[ts_id], self.y[tr_id]
            temp_model = estimator.fit(XTR, YTR)
            y_true_test.extend(self.y[ts_id])
            mean, variance = temp_model.predict(XTS, return_std=True)
            y_pred_test.extend(mean)
            y_pred_test_unc.extend(variance)
            y_true_train.extend(self.y[tr_id])
            mean, variance = temp_model.predict(XTR, return_std=True)
            y_pred_train.extend(mean)
            y_pred_train_unc.extend(variance)
        
        self.y_true_test = np.array(y_true_test)
        self.y_true_train = np.array(y_true_train)
        self.y_pred_test = np.array(y_pred_test)
        self.y_pred_train = np.array(y_pred_train)
        self.y_pred_test_unc = np.array(y_pred_test_unc)
        self.y_pred_train_unc = np.array(y_pred_train_unc)
        
        self.RMSE_test = np.sqrt(mean_squared_error(y_true_test, 
            y_pred_test))
        self.RMSE_train = np.sqrt(mean_squared_error(y_true_train, 
            y_pred_train))
        self.MAE_test = mean_absolute_error(y_true_test, y_pred_test)
        self.MAE_train = mean_absolute_error(y_true_train, y_pred_train)
        self.r2_score_test = r2_score(y_true_test, y_pred_test)
        self.r2_score_train = r2_score(y_true_train, y_pred_train)
        self.final_kernel = estimator.kernel_
        self.log_marginal_likelihood = estimator.log_marginal_likelihood()

    @staticmethod
    def _cook_kernel(recipe):
        K = None
        for key, values in recipe.items():
            kern_lib = import_module('sklearn.gaussian_process.kernels')
            kern = getattr(kern_lib, key)
            if isinstance(values, list):
                if isinstance(values[0], dict):
                    if K is None:
                        K = SKGP._cook_kernel(values[0]) * kern(**values[1])
                    else:
                        K += SKGP._cook_kernel(values[0]) * kern(**values[1])
                else:
                    if K is None:
                        K = values[0] * kern(**values[1])
                    else:
                        K += values[0] * kern(**values[1])
            else:
                if K is None:
                    K = kern(**values)
                else:
                    K += kern(**values)
        return K

    def print_sample_recipe(self):
        print('''
              To cook a kernel, one can follow the guidelines listed below
              to prepare a delicious recipe for a medium rare kernel.
              1.
              Kernel : 100 * RBF(length_scale=100)
              recipe : {'RBF' : [100, {'length_scale' : 100}]}
              2.
              kernel: 100 * RBF(length_scale=100) +
                      White_kernel(noise_laevel=1)
              recipe : {'RBF' : [100, {'length_scale' : 100}],
                        'White_kernel' : {'noise_level' : 1}}
              3.
              kernel: 35**2 * RBF(length_scale=41) *
                      ExpSineSquared(length_scale=2, periodicity=1) +
                      White_kernel(noise_laevel=1)
              recipe : {'ExpSineSquared' : [{'RBF' : [35**2,
                                          {'length_scale' : 41}]},
                                          {'length_scale' : 2,
                                           'Periodicity' : 1},
                        'White_kernel' : {'noise_level' : 1}}
              ''')

    def plot_parity(self, data='train', err_bar=False):
        """A utility function to plot the parity plot along with uncertainties
        of prediction.
        Parameters
        ----------
        data : train | test
        """
        if data not in ['train', 'test']:
            print('data must be either train or test')
            return None
        if data == 'train':
            x = self.y_true_train
            y = self.y_pred_train
            dy = self.y_pred_train_unc
        if data == 'test':
            x = self.y_true_test
            y = self.y_pred_test
            dy = self.y_pred_test_unc
        plt.figure()
        plt.title('Parity plot for {}ing data'.format(data))
        plt.grid(color='b', linestyle='-', linewidth=0.5)
        plt.xlabel("Actual data")
        plt.ylabel("Predicted data")
        if err_bar:
            plt.errorbar(x, y, yerr=dy, fmt='.k', color='g', alpha=0.2)
        plt.scatter(x, y, marker='o', color='g', alpha=0.7,
                 label="ML predicted data")
        plt.plot(x, x, '-', color='black',
                 label="parity line")
        plt.legend(loc='best')

        return plt

