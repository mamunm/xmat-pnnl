from abc import ABC, abstractmethod

class AcquisitionFunction(ABC):

    @abstractmethod
    def __call__(self):
        pass

class LCB(AcquisitionFunction):
    def __init__(self, kappa):
        self.kappa

    def __call__(self, mu, sigma):
        if isinstance(self.kappa, (int, float)):
            k = self.kappa
        else:
            k = self.kappa()
        return mu - k * sigma

class UCB(AcquisitionFunction):
    def __init__(self, kappa):
        self.kappa

    def __call__(self, mu, sigma):
        if isinstance(self.kappa, (int, float)):
            k = self.kappa
        else:
            k = self.kappa()
        return mu + k * sigma

class PI(AcquistionFunction):
    pass

class EI(AcquisitonFunction):
    pass

