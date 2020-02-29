from sklearn.gaussian_process import GaussianProcessRegressor
from .acquisition import UCB, LCB, EI, PI

class Scheduler:
    
    def __init__(self, param_val=None, decay_rate=None):
        self.param_val = param_val
        self.decay_rate = decay_rate
    
    def __call__(self):
        self.param_val -= decay_rate
        return self.param_val



class ActiveLearning:

    def __init__(self, X=None, y=None, 
            acqusition=None, acquisiton_params=None):
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
    

        
        
