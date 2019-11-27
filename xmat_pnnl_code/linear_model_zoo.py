import numpy as np
import pandas as pd
import xmat_pnnl_code as xcode
from itertools import combinations
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class Linear_Model_Zoo:
    def __init__(self, df=None, method='statsmodel'):

        if method not in ['statsmodel', 'sklearn']:
            msg = 'Only statsmodel OLS and '
            msg += 'sklearn LinearRegression is implemented.'
            raise MethodNotImplementedError(msg)
        self.data = df
        self.method = method
    
    @classmethod
    def from_project_name(cls, project_name='9Cr_Data', method='statsmodel'):
        if project_name not in ['9Cr_Data', 'Aus_Steel_Data']:
            msg = 'The requested file does not exist.'
            raise ProjectNotImplementedError(msg)
        if method not in ['statsmodel', 'sklearn']:
            msg = 'Only statsmodel OLS and '
            msg += 'sklearn LinearRegression is implemented.'
            raise MethodNotImplementedError(msg)
        path = '/'.join(xcode.__path__[0].split('/')[:-1])
        if project_name == 'Aus_Steel_Data':
            path += '/data_processing/Aus_Steel_data/Cleaned_data.csv'
        if project_name == '9Cr_Data':
            path += '/data_processing/9Cr_data/Cleaned_data.csv'
        keep_columns = ['ID', 'CT_Temp', 'CT_CS', 'Weighted_AN', 'CT_RT']
        df = pd.read_csv(path)[keep_columns]
        df = df[df.Weighted_AN != 0]
        return cls(df=df, method=method) 

    def data_augmentation(self):
        self.var_cols = ['CT_Temp', 'CT_CS']
        '''
        for i in combinations(['CT_Temp', 'CT_CS'], 2):
            self.data['_'.join(i)] = self.data[list(i)].prod(axis=1)
            self.var_cols.append('_'.join(i))

        func = {
                'log': lambda x: np.log(x),
                'exp': lambda x: np.exp(x),
                'P2': lambda x: x**2,
                'P3': lambda x: x**3,
                '1/P': lambda x: 1/x,
                '1/P2': lambda x: 1/x**2,
                '1/P3': lambda x: 1/x**3
               }
        
        cols_iter = self.var_cols.copy()
        for d in cols_iter:
            for f in ['log', 'exp', 'P2', '1/P', '1/P2']:
                self.data['_'.join([d, f])] = self.data[d].apply(func[f])
                self.var_cols.append('_'.join([d, f]))
        '''

    def get_cleaned_data(self, X, y, n_var):
        
        index = X.notnull().sum(axis=1) == n_var + 1
        X = X[index]
        y = y[index]
        index = np.isfinite(X).sum(axis=1) == n_var + 1
        X = X[index]
        y = y[index]
        
        return X, y

    def build_zoo(self):
        self.zoo = []
        for i in range(1, len(self.var_cols) + 1):
            print('Working on {} of {}.'.format(i, len(self.var_cols)))
            for j in combinations(self.var_cols, i):
                X = self.data[list(j)]
                y = self.data['CT_RT']
                X, y = self.get_cleaned_data(X, y, i)
                if X.empty:
                    continue
                if self.method == 'statsmodel':
                    X = sm.add_constant(X)
                    model = sm.OLS(y, X).fit()
                    self.zoo.append({'descriptors': list(X.columns),
                                     'params': model.params.to_dict(),
                                     'aic': model.aic,
                                     'bic': model.bic,
                                     'Model df': model.df_model,
                                     'n_data': model.nobs,
                                     'R-squared': model.rsquared})
                if self.method == 'sklearn':
                    descriptors = ['const'] + list(X.columns)
                    X = X.to_numpy()
                    y = y.to_numpy()
                    model = LinearRegression().fit(X, y)
                    params = {'const': model.intercept_}
                    for k, v in zip(X.columns, model.coef_):
                        params[k] = v
                    self.zoo.append({'descriptors': descriptors,
                                     'params': params,
                                     'aic': ,
                                     'bic': ,
                                     'Model df': len(X.columns) + 1,
                                     'n_data': len(y),
                                     'R-squared': model.score(X, y)})
    
    @property
    def best_model(self):
        return min(self.zoo, key=lambda x: x['bic'])

    def save_zoo(self, fname='model_zoo.csv'):
        pd.DataFrame(self.zoo).to_csv(fname)

    def plot_envelope(self):
        plt.figure(facecolor='white')
        plt.style.use('classic')
        plt.grid(color='#7366BD', linewidth=0.5, linestyle='--')
        plt.title('BIC Envelope plot')
        plt.xlabel("# of descriptors", fontsize=14)
        plt.ylabel("BIC", fontsize=14)
        data = [(len(v['descriptors']), v['bic']) for v in self.zoo]
        plt.scatter(*zip(*data), color="#1A4876", s=40)
        min_data = []
        unique_descriptors = list(set(i[0] for i in data)).sort()
        for ndes in range(2, unique_descriptors):
            min_data.append(min([i for i in data if i[0]==ndes],
                    key=lambda x: x[1]))
        plt.plot(*zip(*min_data), color="#158078", linewidth=3)
        min_min_data = min(min_data, key=lambda x: x[1])
        plt.xticks([i[0] for i in min_data])
        plt.scatter(min_min_data[0], min_min_data[1],
                color='red', marker='*', s=500)
        plt.tight_layout()
        return plt

if __name__ == "__main__":
    lmz = Linear_Model_Zoo.from_project_name()
    lmz.data_augmentation()
    lmz.build_zoo()
    print(lmz.best_model)
    lmz.save_zoo()
    lmz.plot_envelope().show()

