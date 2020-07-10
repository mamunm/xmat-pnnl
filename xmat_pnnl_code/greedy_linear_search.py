from .sklearn_regression_wrapper import SKREG
import numpy as np

class TopDown:
    
    def __init__(self, df=None, target=None):
        self.df = df
        self.target = target
        self.features = [i for i in self.df.columns if i != self.target]
        X = self.df[self.features].to_numpy()
        y = self.df[self.target].to_numpy().reshape(-1, 1)
        skreg = SKREG(X=X, y=y, estimator='LR', validation='5-Fold')
        skreg.run_reg()
        self.best_pr = skreg.pr_test_mean
        self.history = {}

    def run_search(self):
        while len(self.features) != 0:
            rmse_dict = {}
            for i in self.features:
                fet = [j for j in self.features if j != i]
                X = self.df[fet].to_numpy()
                y = self.df[self.target].to_numpy().reshape(-1, 1)
                skreg = SKREG(X=X, y=y, estimator='LR', validation='5-Fold')
                skreg.run_reg()
                rmse_dict[i] = skreg.pr_test_mean
            f = max(rmse_dict.keys(), key=lambda x: rmse_dict[x])
            v = max(rmse_dict.values())
            self.history[f] = v
            if (v > self.best_pr) or (abs(v - self.best_pr) < 0.01):
                self.best_pr = v
                self.features.remove(f)
            else:
                break

class BottomUp:
    
    def __init__(self, df=None, target=None):
        self.df = df
        self.target = target
        self.features = [i for i in self.df.columns if i != self.target]
        X = self.df[self.features].to_numpy()
        y = self.df[self.target].to_numpy().reshape(-1, 1)
        skreg = SKREG(X=X, y=y, estimator='LR', validation='5-Fold')
        skreg.run_reg()
        self.best_pr = skreg.pr_test_mean
        self.history = {}
        self.b_features = []

    def run_search(self):
        l_f = len(self.features)
        while len(self.b_features) != l_f:
            rmse_dict = {}
            for i in self.features:
                fet = self.b_features + [i]
                X = self.df[fet].to_numpy()
                y = self.df[self.target].to_numpy().reshape(-1, 1)
                skreg = SKREG(X=X, y=y, estimator='LR', validation='5-Fold')
                skreg.run_reg()
                rmse_dict[i] = skreg.pr_test_mean
            f = max(rmse_dict.keys(), key=lambda x: rmse_dict[x])
            v = max(rmse_dict.values())
            self.history[f] = v
            if (v > self.best_pr) or (abs(v - self.best_pr) < 0.01):
                self.best_pr = v
                self.b_features += [f]
            else:
                break
