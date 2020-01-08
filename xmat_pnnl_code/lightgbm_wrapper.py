import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut, KFold
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class LGBM:

    def __init__(self,
            X=None,
            y=None,
            parameters=None,
            cv=5,
            grid_search=False,
            grid_search_scoring='neg_mean_squared_error',
            param_grid=None,
            eval_metric='rmse'):

        self.X = X
        self.y = y
        self.parameters = parameters
        self.cv = cv
        self.grid_search = grid_search
        self.grid_search_scoring = grid_search_scoring
        self.param_grid = param_grid

    def run_model(self):
        if self.grid_search:
            self.run_grid_search()
            self.parameters = self.best_params

        Xtr, Xts, ytr, yts = train_test_split(self.X, self.y, test_size=0.1)

        if self.cv == 'loo':
            cv = LeaveOneOut()
        else:
            cv = KFold(n_splits=self.cv, shuffle=True)

        self.rmse_cv_train = []
        self.r2_cv_train = []
        self.rmse_cv_test = []
        self.r2_cv_test = []
        for n, (tr_id, ts_id) in enumerate(cv.split(ytr)):
            print('Running Validation {} of {}'.format(n, self.cv))
            self.model = lgb.LGBMRegressor(**self.parameters)
            self.model.fit(Xtr[tr_id], ytr[tr_id],
                    eval_set=[(Xtr[ts_id], ytr[ts_id])],
                    eval_metric='rmse', early_stopping_rounds=20)
            y_cv_tr_pred = self.model.predict(Xtr[tr_id],
                    num_iteration=self.model.best_iteration_)
            y_cv_ts_pred = self.model.predict(Xtr[ts_id], 
                    num_iteration=self.model.best_iteration_)
            self.rmse_cv_train.append(np.sqrt(mean_squared_error(
                y_cv_tr_pred, ytr[tr_id])))
            self.rmse_cv_test.append(np.sqrt(mean_squared_error(
                y_cv_ts_pred, ytr[ts_id])))
            self.r2_cv_train.append(linregress(y_cv_tr_pred, 
                ytr[tr_id])[2]**2)
            self.r2_cv_test.append(linregress(y_cv_ts_pred, 
                ytr[ts_id])[2]**2)
        self.rmse_mean_train = np.mean(self.rmse_cv_train)
        self.rmse_std_train = np.std(self.rmse_cv_train)
        self.rmse_mean_test = np.mean(self.rmse_cv_test)
        self.rmse_std_test = np.std(self.rmse_cv_test)
        self.r2_mean_train = np.mean(self.r2_cv_train)
        self.r2_std_train = np.std(self.r2_cv_train)
        self.r2_mean_test = np.mean(self.r2_cv_test)
        self.r2_std_test = np.std(self.r2_cv_test)

        self.model = lgb.LGBMRegressor(**self.parameters)
        self.model.fit(Xtr, ytr, eval_set=[(Xts, yts)], 
                eval_metric='rmse', early_stopping_rounds=20)

    def run_grid_search(self):
        estimator = lgb.LGBMRegressor()
        if self.grid_search_scoring == 'r2':
            self.grid_search_scoring = scoring_func_r2
        gscv = GridSearchCV(estimator, 
                self.param_grid,
                scoring=self.grid_search_scoring,
                cv=self.cv, verbose=20)
        gscv.fit(self.X, self.y)
        self.cv_result = gscv.cv_results_
        self.best_estimator = gscv.best_estimator_
        self.best_score = gscv.best_score_
        self.best_params = gscv.best_params_

    '''
    def plot_feature_importance(self):
        ax = self.model.plot_importance(gbm, 
                importance_type=importance_type,
                max_num_features=10,
                ignore_zero=True, 
                figsize=(12, 8),
                precision=3)
        return plt

    def plot_func_val(self):

    def plot_tree(self):


        

    def predict(self, Xtest):
        self.model.predict(Xtest, num_iteration=self.model.best_iteration_)
    '''

def scoring_func_r2(E, X, y):
    y_pred = E.predict(X)
    return linregress(y, y_pred)[2]**2
