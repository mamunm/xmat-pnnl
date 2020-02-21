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

class GBM_LC:

    def __init__(self,
            package='lightgbm',
            X=None,
            y=None,
            model_scheme=None,
            feature_names=None,
            parameters=None,
            test_size=0.1,
            nrun=5,
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
        self.test_size = test_size
        self.nrun = nrun
        self.CT_Temp = CT_Temp
        self.CT_RT = CT_RT
        self.C = C

    def run_model(self):
        
        est = {'lightgbm': lgb.LGBMRegressor,
               'catboost': catboost.CatBoostRegressor,
               'xgboost': xgboost.XGBRegressor}
        
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
        
        for i in np.arange(self.nrun):
            if self.CT_RT is not None:
                data  = train_test_split(self.X, self.y, self.CT_RT, 
                        self.CT_Temp, self.C, test_size=self.test_size)
                Xtr, Xts, ytr, yts = data[0], data[1], data[2], data[3]
                CT_RTtr, CT_RTts, CT_Temptr = data[4], data[5], data[6]
                CT_Tempts, Ctr, Cts = data[7], data[8], data[9]
                del data
            else:
                Xtr, Xts, ytr, yts = train_test_split(self.X, self.y, 
                        test_size=self.test_size)
            model = est[self.package](**self.parameters)
    
            if self.package == 'lightgbm':
                model.fit(Xtr, ytr, eval_set=[(Xts, yts)],
                    eval_metric='rmse', early_stopping_rounds=20,
                    feature_name=self.feature_names)
            elif self.package == 'xgboost':
                model.fit(Xtr, ytr, eval_set=[(Xts, yts)],
                    eval_metric='rmse', early_stopping_rounds=20)
            else:
                model.fit(Xtr, ytr, eval_set=[(Xts, yts)],
                    early_stopping_rounds=20)
            if self.package == 'lightgbm':
                self.y_cv_tr_pred = model.predict(Xtr, 
                        num_iteration=model.best_iteration_)
                self.y_cv_ts_pred = model.predict(Xts, 
                        num_iteration=model.best_iteration_)
                if self.model_scheme == 'LMP':
                    self.CT_RT_cv_tr_pred = np.exp((
                        self.y_cv_tr_pred*1000/CT_Temptr) - Ctr)
                    self.CT_RT_cv_ts_pred = np.exp((
                        self.y_cv_ts_pred*1000/CT_Tempts) - Cts)
            else:
                self.y_cv_tr_pred = model.predict(Xtr)
                self.y_cv_ts_pred = model.predict(Xts)
                if self.model_scheme == 'LMP':
                    self.CT_RT_cv_tr_pred = np.exp((
                        self.y_cv_tr_pred*1000/CT_Temptr) - Ctr)
                    self.CT_RT_cv_ts_pred = np.exp((
                        self.y_cv_ts_pred*1000/CT_Tempts) - Cts)
            self.y_cv_tr = ytr
            self.y_cv_ts = yts
            if self.CT_RT is not None:
                self.CT_RT_cv_tr = CT_RTtr
                self.CT_RT_cv_ts = CT_RTts
            self.rmse_cv_train.append(np.sqrt(mean_squared_error(
                self.y_cv_tr_pred, ytr)))
            self.rmse_cv_test.append(np.sqrt(mean_squared_error(
                self.y_cv_ts_pred, yts)))
            self.r2_cv_train.append(linregress(self.y_cv_tr_pred, 
                ytr)[2]**2)
            self.r2_cv_test.append(linregress(self.y_cv_ts_pred, 
                yts)[2]**2)
            self.pr_cv_train.append(pr(self.y_cv_tr_pred, ytr))
            self.pr_cv_test.append(pr(self.y_cv_ts_pred, yts))
            if self.CT_RT is not None:
                self.rmse_CT_RT_cv_train.append(np.sqrt(mean_squared_error(
                    self.CT_RT_cv_tr_pred, CT_RTtr)))
                self.rmse_CT_RT_cv_test.append(np.sqrt(mean_squared_error(
                    self.CT_RT_cv_ts_pred, CT_RTts)))
                self.r2_CT_RT_cv_train.append(linregress(self.CT_RT_cv_tr_pred,
                    CT_RTtr)[2]**2)
                self.r2_CT_RT_cv_test.append(linregress(self.CT_RT_cv_ts_pred, 
                    CT_RTts)[2]**2)
                self.pr_CT_RT_cv_train.append(pr(self.CT_RT_cv_tr_pred, 
                    CT_RTtr))
                self.pr_CT_RT_cv_test.append(pr(self.CT_RT_cv_ts_pred, 
                    CT_RTts))
        
        self.N_dp = len(self.y)
        self.N_dp_train = len(ytr)
        self.N_dp_test = len(yts)
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

