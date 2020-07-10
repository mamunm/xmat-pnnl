import numpy as np
import pandas as pd
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
from .autoencoder import AutoEncoder
import sys

class VaeGBM:

    def __init__(self,
            package='lightgbm',
            X=None,
            scale=None,
            features=None,
            element=None,
            parameters=None,
            cv=5,
            gen_n=None,
            gen_per_direction=None,
            eval_metric='rmse',
            vae_arch=None,
            vae_loss=None,
            vae_epochs=None):

        self.package = package
        self.X = X
        self.scale = scale
        self.features = features
        self.element = element
        self.parameters = parameters
        self.cv = cv
        if vae_arch is not None:
            self.vae_arch = vae_arch
        else:
            self.vae_arch = [self.X.shape[1], 12, 6, 2]
        self.gen_n = gen_n
        self.gen_per_direction = gen_per_direction
        self.scale = scale
        self.vae_loss = vae_loss
        self.vae_epochs = vae_epochs


    def run_model(self):

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
        
        est = {'lightgbm': lgb.LGBMRegressor,
               'catboost': catboost.CatBoostRegressor,
               'xgboost': xgboost.XGBRegressor}
        
        self.model = []
        self.gen_sample = []
        model = est[self.package](**self.parameters)
        for n, (tr_id, ts_id) in enumerate(cv.split(self.X)):
            print('Running Validation {} of {}'.format(n, self.cv))
            ae = AutoEncoder(arch=self.vae_arch, X=self.X[tr_id], 
                loss='xent', epochs=2000)

            ae.build_model()
            X_gen = None
            while X_gen is None:
                if self.gen_n is not None:
                    X_gen = self.validate_xgen(generated_X=
                            ae.get_random_alloy(n_samples=self.gen_n))
                if self.gen_per_direction is not None:
                    X_gen = self.validate_xgen(generated_X=
                            ae.get_linspace_alloy(n_range=(-3, 3),
                                n_sample_per_direction=self.gen_per_direction))
            self.gen_sample.append(self.scale.inverse_transform(X_gen))
            y_gen = self.scale.inverse_transform(X_gen)[:, -1]
            y_orig = self.scale.inverse_transform(self.X[tr_id])[:, -1]
            y_ts = self.scale.inverse_transform(self.X[ts_id])[:, -1]
            X_gen = X_gen[:, :-1]
            X_orig = self.X[tr_id][:, :-1]
            X_tr = np.vstack([X_orig, X_gen])
            y_tr = np.concatenate([y_orig, y_gen])
            if self.package == 'lightgbm':
                self.model.append(model.fit(X_tr, y_tr,
                    eval_set=[(self.X[ts_id][:, :-1], y_ts)],
                    eval_metric='rmse', early_stopping_rounds=20,
                    feature_name=self.feature_names))
            elif self.package == 'xgboost':
                self.model.append(model.fit(X_tr, y_tr,
                    eval_set=[(self.X[ts_id][:, :-1], y_ts)],
                    eval_metric='rmse', early_stopping_rounds=20))
            else:
                self.model.append(model.fit(X_tr, y_tr,
                    eval_set=[(self.X[ts_id][:, :-1], y_ts)],
                    early_stopping_rounds=20))
            if self.package == 'lightgbm':
                self.y_cv_tr_pred = self.model[-1].predict(X_tr,
                    num_iteration=self.model[-1].best_iteration_)
                self.y_cv_ts_pred = self.model[-1].predict(
                        self.X[ts_id][:, :-1], num_iteration=
                        self.model[-1].best_iteration_)
            else:
                self.y_cv_tr_pred = self.model[-1].predict(
                        self.X[tr_id][:, :-1])
                self.y_cv_ts_pred = self.model[-1].predict(
                        self.X[ts_id][:, :-1])
            self.y_cv_tr = y_orig
            self.y_cv_ts = y_ts
            self.rmse_cv_train.append(np.sqrt(mean_squared_error(
                self.y_cv_tr_pred, self.y_cv_tr)))
            self.rmse_cv_test.append(np.sqrt(mean_squared_error(
                self.y_cv_ts_pred, self.y_cv_ts)))
            self.r2_cv_train.append(linregress(self.y_cv_tr_pred, 
                self.y_cv_tr)[2]**2)
            self.r2_cv_test.append(linregress(self.y_cv_ts_pred, 
                self.y_cv_ts)[2]**2)
            self.pr_cv_train.append(pr(self.y_cv_tr_pred, self.y_cv_tr))
            self.pr_cv_test.append(pr(self.y_cv_ts_pred, self.y_cv_ts))
        
            self.N_dp = len(self.X[:, -1])
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

    def validate_xgen(self, generated_X=None):
        df = pd.DataFrame(self.scale.inverse_transform(generated_X), 
                columns=self.features)
        df = df[abs(df[self.element].sum(axis=1)-100) < 0.05]
        return self.scale.transform(df.to_numpy())

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
