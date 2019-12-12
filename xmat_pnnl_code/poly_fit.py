from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

class PolyFit:

    def __init__(self, df=None, target=None, degree=2):
        self.df = df
        self.target = target
        self.degree = degree

    def fit(self):
        polyfeatures = PolynomialFeatures(degree=self.degree)
        X = self.df[[i for i in self.df.columns if i!=self.target]]
        self.X = polyfeatures.fit_transform(X.to_numpy())
        self.y = self.df[self.target].to_numpy()
        self.model = LinearRegression().fit(self.X, self.y)
        self.y_pred = self.model.predict(self.X)
        self.score = self.model.score(self.X, self.y)

    def plot_model(self, text=None):
        xn, yn = zip(*sorted(zip(self.df['LMP'].to_numpy(), self.y_pred)))
        plt.plot(xn, yn, color='#1A4876', linewidth=4)
        plt.scatter(self.df['LMP'].to_numpy(), self.y, color='#158078',
                marker='h', s=50)
        plt.title('Polynomial model with degree={}'.format(self.degree))
        plt.xlabel('LMP')
        plt.ylabel('{}'.format(self.target))
        if text:
            ax = plt.gca()
            plt.text(0.6, 0.9, text, size=10,
                     ha="center", va="center",
                     transform=ax.transAxes)
        return plt


