import numpy as np
import lightgbm as lgb
import catboost 
import xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut, KFold
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import pearsonr as pr

class GBM:

    def __init__(self,
            package='lightgbm',
            X=None,
            y=None,
            model_scheme=None,
            feature_names=None,
            parameters=None,
            cv=5,
            test_size=0.1,
            grid_search=False,
            grid_search_scoring='neg_mean_squared_error',
            param_grid=None,
            eval_metric='rmse',
            CT_Temp=None,
            CT_RT=None,
            C=None):

        self.package = package
        self.X = X
        self.y = y
        self.model_scheme = model_scheme
        self.feature_names = feature_names
        self.parameters = parameters
        self.cv = cv
        self.test_size = test_size
        self.grid_search = grid_search
        self.grid_search_scoring = grid_search_scoring
        self.param_grid = param_grid
        self.CT_Temp = CT_Temp
        self.CT_RT = CT_RT
        self.C = C

    def run_model(self):
        if self.grid_search:
            self.run_grid_search()
            self.parameters = self.best_params
        
        '''
        Xtr, Xts, ytr, yts, CT_RT_tr, CT_RT_ts = train_test_split(self.X, 
                self.y, self.CT_RT, test_size=self.test_size)
        '''

        if self.cv == 'loo':
            cv = LeaveOneOut()
        else:
            cv = KFold(n_splits=self.cv, shuffle=True)

        self.rmse_cv_train = []
        self.r2_cv_train = []
        self.rmse_cv_test = []
        self.r2_cv_test = []
        self.pr_cv_train = []
        self.pr_cv_test = []
        if self.CT_RT is not None:
            self.rmse_CT_RT_cv_train = []
            self.r2_CT_RT_cv_train = []
            self.rmse_CT_RT_cv_test = []
            self.r2_CT_RT_cv_test = []
            self.pr_CT_RT_cv_train = []
            self.pr_CT_RT_cv_test = []
        
        est = {'lightgbm': lgb.LGBMRegressor,
               'catboost': catboost.CatBoostRegressor,
               'xgboost': xgboost.XGBRegressor}
        
        self.model = []
        model = est[self.package](**self.parameters)
        for n, (tr_id, ts_id) in enumerate(cv.split(self.y)):
            print('Running Validation {} of {}'.format(n, self.cv))
            if self.package == 'lightgbm':
                self.model.append(model.fit(self.X[tr_id], self.y[tr_id],
                    eval_set=[(self.X[ts_id], self.y[ts_id])],
                    eval_metric='rmse', early_stopping_rounds=20,
                    feature_name=self.feature_names))
            elif self.package == 'xgboost':
                self.model.append(model.fit(self.X[tr_id], self.y[tr_id],
                    eval_set=[(self.X[ts_id], self.y[ts_id])],
                    eval_metric='rmse', early_stopping_rounds=20))
            else:
                self.model.append(model.fit(self.X[tr_id], self.y[tr_id],
                    eval_set=[(self.X[ts_id], self.y[ts_id])],
                    early_stopping_rounds=20))
            if self.package == 'lightgbm':
                self.y_cv_tr_pred = self.model[-1].predict(self.X[tr_id],
                    num_iteration=self.model[-1].best_iteration_)
                self.y_cv_ts_pred = self.model[-1].predict(self.X[ts_id], 
                    num_iteration=self.model[-1].best_iteration_)
                if self.model_scheme == 'LMP':
                    self.CT_RT_cv_tr_pred = np.exp((
                        self.y_cv_tr_pred*1000/self.CT_Temp[tr_id]) 
                        - self.C[tr_id])
                    self.CT_RT_cv_ts_pred = np.exp((
                        self.y_cv_ts_pred*1000/self.CT_Temp[ts_id]) 
                        - self.C[ts_id])
            else:
                self.y_cv_tr_pred = self.model[-1].predict(self.X[tr_id])
                self.y_cv_ts_pred = self.model[-1].predict(self.X[ts_id])
                if self.model_scheme == 'LMP':
                    self.CT_RT_cv_tr_pred = np.exp((
                        self.y_cv_tr_pred*1000/self.CT_Temp[tr_id]) 
                        - self.C[tr_id])
                    self.CT_RT_cv_ts_pred = np.exp((
                        self.y_cv_ts_pred*1000/self.CT_Temp[ts_id]) 
                        - self.C[ts_id])
            self.y_cv_tr = self.y[tr_id]
            self.y_cv_ts = self.y[ts_id]
            if self.CT_RT is not None:
                self.CT_RT_cv_tr = self.CT_RT[tr_id]
                self.CT_RT_cv_ts = self.CT_RT[ts_id]
            self.rmse_cv_train.append(np.sqrt(mean_squared_error(
                self.y_cv_tr_pred, self.y[tr_id])))
            self.rmse_cv_test.append(np.sqrt(mean_squared_error(
                self.y_cv_ts_pred, self.y[ts_id])))
            self.r2_cv_train.append(linregress(self.y_cv_tr_pred, 
                self.y[tr_id])[2]**2)
            self.r2_cv_test.append(linregress(self.y_cv_ts_pred, 
                self.y[ts_id])[2]**2)
            self.pr_cv_train.append(pr(self.y_cv_tr_pred, self.y[tr_id]))
            self.pr_cv_test.append(pr(self.y_cv_ts_pred, self.y[ts_id]))
            if self.CT_RT is not None:
                self.rmse_CT_RT_cv_train.append(np.sqrt(mean_squared_error(
                    self.CT_RT_cv_tr_pred, self.CT_RT[tr_id])))
                self.rmse_CT_RT_cv_test.append(np.sqrt(mean_squared_error(
                    self.CT_RT_cv_ts_pred, self.CT_RT[ts_id])))
                self.r2_CT_RT_cv_train.append(linregress(self.CT_RT_cv_tr_pred,
                    self.CT_RT[tr_id])[2]**2)
                self.r2_CT_RT_cv_test.append(linregress(self.CT_RT_cv_ts_pred, 
                    self.CT_RT[ts_id])[2]**2)
                self.pr_CT_RT_cv_train.append(pr(self.CT_RT_cv_tr_pred, 
                    self.CT_RT[tr_id]))
                self.pr_CT_RT_cv_test.append(pr(self.CT_RT_cv_ts_pred, 
                    self.CT_RT[ts_id]))
        
        self.N_dp = len(self.y)
        self.rmse_mean_train = np.mean(self.rmse_cv_train)
        self.rmse_std_train = np.std(self.rmse_cv_train)
        self.rmse_mean_test = np.mean(self.rmse_cv_test)
        self.rmse_std_test = np.std(self.rmse_cv_test)
        self.r2_mean_train = np.mean(self.r2_cv_train)
        self.r2_std_train = np.std(self.r2_cv_train)
        self.r2_mean_test = np.mean(self.r2_cv_test)
        self.r2_std_test = np.std(self.r2_cv_test)
        self.pr_mean_train = np.mean([i[0] for i in self.pr_cv_train])
        self.pr_std_train = np.std([i[0] for i in self.pr_cv_train])
        self.pr_mean_test = np.mean([i[0] for i in self.pr_cv_test])
        self.pr_std_test = np.std([i[0] for i in self.pr_cv_test])
        if self.CT_RT is not None:
            self.rmse_CT_RT_mean_train = np.mean(self.rmse_CT_RT_cv_train)
            self.rmse_CT_RT_std_train = np.std(self.rmse_CT_RT_cv_train)
            self.rmse_CT_RT_mean_test = np.mean(self.rmse_CT_RT_cv_test)
            self.rmse_CT_RT_std_test = np.std(self.rmse_CT_RT_cv_test)
            self.r2_CT_RT_mean_train = np.mean(self.r2_CT_RT_cv_train)
            self.r2_CT_RT_std_train = np.std(self.r2_CT_RT_cv_train)
            self.r2_CT_RT_mean_test = np.mean(self.r2_CT_RT_cv_test)
            self.r2_CT_RT_std_test = np.std(self.r2_CT_RT_cv_test)
            self.pr_CT_RT_mean_train = np.mean([i[0] 
                for i in self.pr_CT_RT_cv_train])
            self.pr_CT_RT_std_train = np.std([i[0] 
                for i in self.pr_CT_RT_cv_train])
            self.pr_CT_RT_mean_test = np.mean([i[0] 
                for i in self.pr_CT_RT_cv_test])
            self.pr_CT_RT_std_test = np.std([i[0] 
                for i in self.pr_CT_RT_cv_test])

        '''
        self.model = est[self.package](**self.parameters)
        if self.package == 'lightgbm':
            self.model.fit(Xtr, ytr, eval_set=[(Xts, yts)], 
                eval_metric='rmse', early_stopping_rounds=20,
                feature_name=self.feature_names)
        elif self.package == 'xgboost':
            self.model.fit(Xtr, ytr, eval_set=[(Xts, yts)], 
                eval_metric='rmse', early_stopping_rounds=20)
        else:
            self.model.fit(Xtr, ytr, eval_set=[(Xts, yts)], 
                early_stopping_rounds=20)
        '''
    
    def parity_plot(self, data='train', quantity='LMP', scheme=2):
        """A utility function to plot the parity between predicted and
        actual values.
        Parameters
        ----------
        data : train | test
        """
        if data not in ['train', 'test']:
            print('data must be either train or test')
            return None
        if data == 'train' and quantity == 'LMP' and scheme != 1:
            x = self.y_cv_tr
            y = self.y_cv_tr_pred
        if data == 'test' and quantity == 'LMP' and scheme != 1:
            x = self.y_cv_ts
            y = self.y_cv_ts_pred
        if data == 'train' and quantity == 'CT_RT' and scheme != 1:
            x = self.CT_RT_cv_tr
            y = self.CT_RT_cv_tr_pred
        if data == 'test' and quantity == 'CT_RT' and scheme != 1:
            x = self.CT_RT_cv_ts
            y = self.CT_RT_cv_ts_pred
        if data == 'train' and quantity == 'CT_RT' and scheme == 1:
            x = self.y_cv_tr
            y = self.y_cv_tr_pred
        if data == 'test' and quantity == 'CT_RT' and scheme == 1:
            x = self.y_cv_ts
            y = self.y_cv_ts_pred
        plt.figure(figsize=(10, 8))
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


    def run_grid_search(self):
        est = {'lightgbm': lgb.LGBMRegressor,
               'catboost': catboost.CatBoostRegressor,
               'xgboost': xgboost.XGBRegressor}
        estimator = est[self.package]()
        if self.grid_search_scoring == 'r2':
            self.grid_search_scoring = scoring_func_r2
        gscv = GridSearchCV(estimator, 
                self.param_grid,
                scoring=self.grid_search_scoring,
                cv=self.cv, verbose=20, n_jobs=-1)
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
