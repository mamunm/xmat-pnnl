import numpy as np
from sklearn.metrics import (mean_absolute_error, 
        mean_squared_error, r2_score)
from importlib import import_module
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from scipy.stats import linregress

class SKGridReg:
    '''An object to oversee a grid search CV over the parameter space for a 
    sklearn regression run and store all the information pertinent
    to that run as a self contained source.'''

    def __init__(self,
                X=None,
                y=None,
                estimator=None,
                estimator_param_space=None,
                cv=5,
                scoring='r2'):

        self.X = X
        self.y = y
        self.estimator = estimator
        self.estimator_param_space = estimator_param_space
        self.cv = cv
        self.scoring = scoring

    def run_grid_search(self):

        if self.estimator == 'MLP':
            est = import_module('sklearn.neural_network')
            estimator = getattr(est, 'MLPRegressor')()
        if self.estimator == 'RF':
            est = import_module('sklearn.ensemble')
            estimator = getattr(est, 'RandomForestRegressor')()

        if self.scoring == 'r2':
            self.scoring = scoring_func

        gscv = GridSearchCV(estimator, 
                            self.estimator_param_space,
                            scoring=scoring_func,
                            cv=self.cv,
                            verbose=20)
        gscv.fit(self.X, self.y)
        self.cv_result = gscv.cv_results_
        self.best_estimator = gscv.best_estimator_
        self.best_score = gscv.best_score_
        self.best_params = gscv.best_params_

def scoring_func(E, X, y):
    y_pred = E.predict(X)
    return linregress(y, y_pred)[2]**2
