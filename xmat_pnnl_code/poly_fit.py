import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

class PolyFit:

    def __init__(self, df=None, target=None, degree=2, logify_y=False):
        self.df = df
        self.target = target
        self.degree = degree
        self.logify_y = logify_y

    def fit(self):
        self.descriptors = [i for i in self.df.columns if i!=self.target]
        if self.degree > 1:
            polyfeatures = PolynomialFeatures(degree=self.degree)
            X = self.df[self.descriptors]
            self.X = polyfeatures.fit_transform(X.to_numpy())
        else:
            self.X = self.df[self.descriptors].to_numpy()
        self.y = self.df[self.target].to_numpy()
        if self.logify_y:
            self.y = np.log(self.y)
        self.model = LinearRegression().fit(self.X, self.y)
        self.y_pred = self.model.predict(self.X)
        self.score = self.model.score(self.X, self.y)

    def plot_model(self, text=None):
        if self.logify_y:
            xn, yn = zip(*sorted(zip(self.df['LMP'].to_numpy(), 
                self.y_pred)))
        else:
            xn, yn = zip(*sorted(zip(self.df['LMP'].to_numpy(), self.y_pred)))
        plt.plot(xn, yn, color='#1A4876', linewidth=4)
        if self.logify_y:
            plt.scatter(self.df['LMP'].to_numpy(), 
                    self.y, color='#158078', marker='h', s=50)
        else:
            plt.scatter(self.df['LMP'].to_numpy(), 
                    self.y, color='#158078', marker='h', s=50)
        plt.title('Polynomial model with degree={}'.format(self.degree))
        plt.xlabel('LMP')
        if self.logify_y:
            plt.ylabel('log({})'.format(self.target))
        else:
            plt.ylabel('{}'.format(self.target))
        ax = plt.gca()
        text = 'r2 score: {}'.format(self.score)
        plt.text(0.6, 0.9, text, size=10,
                 ha="center", va="center",
                 transform=ax.transAxes)
        return plt


