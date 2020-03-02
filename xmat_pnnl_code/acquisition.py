from abc import ABC, abstractmethod

class AcquisitionFunction(ABC):

    @abstractmethod
    def __call__(self):
        pass

class Scheduler:

    def __init__(self, param_val=None, decay=None):
        self.param_val = param_val
        self.decay = decay

    def __call__(self):
        self.param_val -= decay
        return self.param_val

class LCB(AcquisitionFunction):
    def __init__(self, kappa=1, decay=0):
        self.kappa = Scheduler(param_val=kappa, decay=decay)

    def __call__(self, mu, sigma):
        k = self.kappa()
        return mu - k * sigma

class UCB(AcquisitionFunction):
    def __init__(self, kappa=1, decay=decay):
        self.kappa = Scheduler(param_val=kappa, decay=decay)

    def __call__(self, mu, sigma):
        k = self.kappa()
        return mu + k * sigma

class PI(AcquisitionFunction):
    pass

class EI(AcquisitionFunction):
    pass

