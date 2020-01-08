import numpy as np
import pandas as pd
from xmat_pnnl_code import PolyFit
from scipy.optimize import fmin
from sklearn.preprocessing import PolynomialFeatures

class ConstantMatcher:

    def __init__(self, 
                 df=None, 
                 stress=None,
                 temp=None,
                 RT=None,
                 groupby=None,
                 ID_list=None,
                 degree=2,
                 logify_y=False):

        self.df = df
        self.stress = stress
        self.temp = temp
        self.RT = RT
        self.groupby = groupby
        self.ID_list = ID_list
        self.degree = degree
        self.logify_y = logify_y

    def get_C(self):
        self.C_dict = {}
        if self.groupby:
            for n, (k, g) in enumerate(self.df.groupby(self.groupby)):
                if n == 0:
                    self.C_dict[k] = 25
                    g['LMP'] = 1e-3 * (g[self.temp]) * (
                            np.log(g[self.RT]) + 25)
                    poly = PolyFit(df=g[['LMP', self.stress]], 
                            target=self.stress, 
                            degree=self.degree,
                            logify_y=self.logify_y)
                    poly.fit()
                    model = poly.model
                else:
                    self.C_dict[k] = self.maximize_score(g, model)
        
        if self.ID_list:
            for n, alloy_id in enumerate(self.ID_list):
                if n == 0:
                    g = self.df[self.df['ID'] == alloy_id]
                    g['LMP'] = 1e-3 * (g[self.temp]) * (
                            np.log(g[self.RT]) + 25)
                    poly = PolyFit(df=g[['LMP', self.stress]], 
                            target='CT_CS', 
                            degree=self.degree,
                            logify_y=self.logify_y)
                    poly.fit()
                    model = poly.model
                    self.C_dict[alloy_id] = {'C': 25, 'score':poly.score}
                else:
                    g = self.df[self.df['ID'] == alloy_id]
                    C, score = self.maximize_score(g, model)
                    self.C_dict[alloy_id] = {'C': C, 'score': score} 
        return self.C_dict

    def maximize_score(self, df, model):
        C0 = 25
        C =  fmin(self.compute_score, C0, ftol=1e-4, 
                args=(df, model), full_output=1)
        return C[0][0], -C[1]
    
    def compute_score(self, C, *args):
        df = args[0]
        model = args[1]
        df['LMP'] = 1e-3 * (df[self.temp]) * (np.log(df[self.RT]) + C)
        if not self.degree == 1:
            X = PolynomialFeatures(degree=self.degree).fit_transform(
                    df[['LMP']].to_numpy())
        else:
            X = df[['LMP']].to_numpy()
        return -model.score(X, np.log(df[self.stress]))

