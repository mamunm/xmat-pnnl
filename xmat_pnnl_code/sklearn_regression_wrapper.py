import numpy as np
from sklearn.metrics import (mean_absolute_error, 
        mean_squared_error)
from scipy.stats import linregress
from importlib import import_module
import matplotlib.pyplot as plt

class SKREG:
    '''An object to oversee a sklearn regression run and store all the 
    information pertinent to that run as a self contained source.'''

    def __init__(self,
                X=None,
                y=None,
                estimator=None,
                estimator_param=None,
                validation=None):

        self.X = X
        self.y = y
        self.estimator = estimator
        self.estimator_param = estimator_param
        self.validation = validation

    def run_reg(self):

        if self.estimator == 'MLP':
            est = import_module('sklearn.neural_network')
            estimator = getattr(est, 'MLPRegressor')
        if self.estimator == 'LR':
            est = import_module('sklearn.linear_model')
            estimator = getattr(est, 'LinearRegression')
        if self.estimator == 'RF':
            est = import_module('sklearn.ensemble')
            estimator = getattr(est, 'RandomForestRegressor')

        if not self.estimator_param:
            estimator = estimator()
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
            
        self.rmse_train = []
        self.rmse_test = []
        self.mae_train = []
        self.mae_test = []
        self.r2_train = []
        self.r2_test = []
        self.y_true_train = []
        self.y_pred_train = []
        self.y_true_test = []
        self.y_pred_test = []
        for n, (tr_id, ts_id) in enumerate(val.split(self.y)):
            print('Running validation model no. {}'.format(n+1))
            XTR, XTS, YTR = self.X[tr_id], self.X[ts_id], self.y[tr_id]
            temp_model = estimator.fit(XTR, YTR)
            y_true = self.y[ts_id]
            y_pred = temp_model.predict(XTS)
            y_pred_train = temp_model.predict(XTR)
            self.y_true_train.extend(YTR)
            self.y_pred_train.extend(y_pred_train)
            self.y_true_test.extend(y_true)
            self.y_pred_test.extend(y_pred)
            self.rmse_train.append(np.sqrt(mean_squared_error(y_pred_train, 
                YTR)))
            self.rmse_test.append(np.sqrt(mean_squared_error(y_pred, y_true)))
            self.mae_train.append(mean_absolute_error(y_pred_train, YTR))
            self.mae_test.append(mean_absolute_error(y_pred, y_true))
            self.r2_train.append(linregress(y_pred_train, YTR)[2]**2)
            self.r2_test.append(linregress(y_pred, y_true)[2]**2)

        self.rmse_train_mean = np.mean(self.rmse_train)
        self.rmse_train_std = np.std(self.rmse_train)
        self.mae_train_mean = np.mean(self.mae_train)
        self.mae_train_std = np.std(self.mae_train)
        self.r2_train_mean = np.mean(self.r2_train)
        self.r2_train_std = np.std(self.r2_train)
        
        self.rmse_test_mean = np.mean(self.rmse_test)
        self.rmse_test_std = np.std(self.rmse_test)
        self.mae_test_mean = np.mean(self.mae_test)
        self.mae_test_std = np.std(self.mae_test)
        self.r2_test_mean = np.mean(self.r2_test)
        self.r2_test_std = np.std(self.r2_test)

    def plot_parity(self, data='train'):
        """A utility function to plot the parity between predicted and 
        actual values.
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
        if data == 'test':
            x = self.y_true_test
            y = self.y_pred_test
        plt.figure()
        plt.title('Parity plot for {}ing data'.format(data))
        plt.grid(color='b', linestyle='-', linewidth=0.5)
        plt.xlabel("Actual data")
        plt.ylabel("Predicted data")
        plt.scatter(x, y, marker='o', color='g', alpha=0.7,
                 label="ML predicted data")
        plt.plot(x, x, '-', color='black',
                 label="parity line")
        plt.legend(loc='best')

        return plt

