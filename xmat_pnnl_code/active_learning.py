from sklearn.gaussian_process import GaussianProcessRegressor
from .acquisition import UCB, LCB, EI, PI

class ActiveLearning:

    def __init__(self, X=None, y=None, 
            acqusition=None, acquisiton_params=None,
            stopping_rounds=10):
        self.X = X
        self.y = y
        if acquisition not in ['UCB', 'LCB', 'EI', 'PI']:
            raise ValueError('Not a valid acquisiton function!')
        if acqusition == 'UCB':
            self.acquisition = UCB(**acquisition_params)
        if acqusition == 'LCB':
            self.acquisition = LCB(**acquisition_params)
        if acqusition == 'EI':
            self.acquisition = EI(**acquisition_params)
        if acqusition == 'PI':
            self.acquisition = PI(**acquisition_params)

    def run_model(self):
        pass
        
    

        
        
