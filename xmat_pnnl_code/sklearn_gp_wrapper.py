import numpy as np
from sklearn.metrics import (mean_absolute_error, 
        mean_squared_error)
from scipy.stats import linregress
from scipy.stats import pearsonr
from importlib import import_module
import matplotlib.pyplot as plt

class SKGP:
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
        
        print('Fitting the master model. Hang tight!')
        self.model = estimator.fit(self.X, self.y)
        #Model Validation
        print('Initializing validation.')
        if self.validation == 'leave_one_out':
            val = getattr(import_module('sklearn.model_selection'), 
                    'LeaveOneOut')()
        else:
            val = getattr(import_module('sklearn.model_selection'), 
                    'KFold')(n_splits=int(self.validation.split('-')[0]))
            
        self.RMSE_test_cv = []
        self.RMSE_train_cv = []
        self.PCC_train_cv = []
        self.PCC_test_cv = []
        for n, (tr_id, ts_id) in enumerate(val.split(self.y)):
            print('Running validation model no. {}'.format(n+1))
            temp_model = estimator.fit(self.X[tr_id], self.y[tr_id])
            YTR_pred, YTR_pred_unc = temp_model.predict(
                    self.X[tr_id], return_std=True)
            YTS_pred, YTS_pred_unc = temp_model.predict(
                    self.X[ts_id], return_std=True)
            self.RMSE_train_cv.append(np.sqrt(mean_squared_error(
                YTR_pred, self.y[tr_id])))
            self.RMSE_test_cv.append(np.sqrt(mean_squared_error(
                YTS_pred, self.y[ts_id])))
            self.PCC_train_cv.append(pearsonr(YTR_pred, self.y[tr_id]))
            self.PCC_test_cv.append(pearsonr(YTS_pred, self.y[ts_id]))
        
        self.y_true_test = np.array(self.y[ts_id])
        self.y_true_train = np.array(self.y[tr_id])
        self.y_pred_test = np.array(YTS_pred)
        self.y_pred_train = np.array(YTR_pred)
        self.y_pred_test_unc = np.array(YTS_pred_unc)
        self.y_pred_train_unc = np.array(YTR_pred_unc)
        
        self.RMSE_train_mean = np.mean(self.RMSE_train_cv)
        self.RMSE_train_std = np.std(self.RMSE_train_cv)
        self.RMSE_test_mean = np.mean(self.RMSE_test_cv)
        self.RMSE_test_std = np.std(self.RMSE_test_cv)
        self.PCC_train_mean = np.mean([i[0] for i in self.PCC_train_cv if i])
        self.PCC_train_std = np.std([i[0] for i in self.PCC_train_cv if i])
        self.PCC_test_mean = np.mean([i[0] for i in self.PCC_test_cv if i])
        self.PCC_test_std = np.std([i[0] for i in self.PCC_test_cv if i])
        
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

