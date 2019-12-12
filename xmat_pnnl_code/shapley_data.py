from itertools import combinations
from xmat_pnnl_code import PolyFit

class Shapley:

    def __init__(self, df=None, target=None, degree=2):
        self.df = df
        self.target = target
        self.degree = degree

    def get_phi(self):
        phi = []
        
        for n, i in enumerate(self.df.index):
            score_lib = {}
            print('Working on {} number data points of {}'.format(
                n, len(self.df.index)))
            score_comb = []
            ind = [j for j in self.df.index if j != i]
            for j in range(0, len(ind)+1):
                score = []
                for k in combinations(ind, j):
                    key_exclude = '_'.join(sorted([str(z) for z in k]))
                    key_include = '_'.join(sorted([str(z) for z in k] + [str(i)]))
                    if j != 0:
                        if key_exclude in score_lib:
                            score_exclude = score_lib[key_exclude]
                        else:
                            poly_exclude = PolyFit(df=self.df.loc[list(k)], 
                                    target=self.target, degree=2)
                            poly_exclude.fit()
                            score_exclude = poly_exclude.score
                            score_lib[key_exclude] = score_exclude
                    else:
                        score_exclude = 0
                    if key_include in score_lib:
                        score_include = score_lib[key_include]
                    else:
                        poly_include = PolyFit(df=self.df.loc[list(k) + [i]], 
                                    target=self.target, degree=2)
                        poly_include.fit()
                        score_include = poly_include.score
                        score_lib[key_include] = score_include
                    score.append(score_include - score_exclude)
                score_comb.append(sum(score)/len(score))
            phi.append(sum(score_comb)/len(score_comb))
        
        phi_percentage = [i/sum(phi)*100 for i in phi]
        
        return phi, phi_percentage

