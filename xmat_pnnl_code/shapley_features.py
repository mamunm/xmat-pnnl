from itertools import combinations
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, WhiteKernel,
        DotProduct, Matern)
from sklearn.metrics import r2_score

class ShapleyFeatures:
    
    def __init__(self, 
            df=None,
            target=None,
            features=None):
        self.df = df
        self.target = target
        self.features = features

    def get_phi(self):
        kernel = 1.0*RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        kernel += 1.0*DotProduct(sigma_0=1.0) + 1.0*Matern(length_scale=1.0)
        estimator = GaussianProcessRegressor(kernel=kernel, 
                n_restarts_optimizer=8, alpha=0)
        phi = []
        score_lib = {} 
        for n, i in enumerate(self.features):
            print('Working on {}'.format(i))
            score_comb = []
            f = [j for j in self.features if j != i]
            for j in range(0, len(f)+1):
                score = []
                for k in combinations(f, j):
                    if j != 0:
                        key = '_'.join(sorted(list(k)))
                        if key in score_lib:
                            score_exclude = score_lib[key]
                            print('Getting data from lib.')
                        else:
                            X = self.df[list(k)]
                            y = self.df[self.target]
                            estimator.fit(X, y)
                            score_exclude = r2_score(estimator.predict(X), y)
                            score_lib[key] = score_exclude
                            msg = 'Adding {} with score {} to the score lib'
                            print(msg.format(key, score_exclude))
                    else:
                        score_exclude = 0
                    key = '_'.join(sorted(list(k)+[i]))
                    if key in score_lib:
                        score_include = score_lib[key]
                        print('Getting data from lib.')
                    else:
                        X = self.df[list(k) + [i]]
                        y = self.df[self.target]
                        estimator.fit(X, y) 
                        score_include = r2_score(estimator.predict(X), y)
                        score_lib[key] = score_include
                        msg = 'Adding {} with score {} to the score lib'
                        print(msg.format(key, score_include))
                    score.append(score_include - score_exclude)
                score_comb.append(sum(score)/len(score))
            phi.append(sum(score_comb)/len(score_comb))

        phi_percentage = [i/sum(phi)*100 for i in phi]

        return {'features': self.features,
                'phi': phi, 
                'phi_percentage': phi_percentage,
                'score_lib': score_lib}

