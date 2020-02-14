import numpy as np
from sklearn.impute import SimpleImputer
import sklearn.preprocessing as skpre
from sklearn.decomposition import PCA

class ProcessData():
    """ An object to hold the data for preprocessing and later it will return
    the processsed data when needed
    TODO: Metadata modification
    """

    def __init__(self,
                 X = None,
                 y = None,
                 y2 = None,
                 features=None,
                 metadata=None):
        self.X = X
        self.y = y
        self.y2 = y2
        self.features = features
        self.metadata=metadata

    def get_data(self):
        """Returns the data at any point it was called"""
        return {'X': self.X, 
                'y': self.y, 
                'y2': self.y2,
                'features': self.features, 
                'metadata': self.metadata}

    def remove_instance(self, null_count=0.5):
        """function to remove any data instance with more than
        null_count * 100% null values"""

        mask = [i for i, XX in enumerate(self.X)
                if np.isnan(XX).mean() < null_count]
        self.X = self.X[mask]
        self.y = self.y[mask]
        if self.y2 is not None:
            self.y2 = [self.y2[i] for i in mask]
        if self.metadata is not None:
            self.metadata = [self.metadata[i] for i in mask]

    def remove_null_features(self, null_count=0.5):
        """function to remove any feature with more than
        null_count * 100% null values"""

        mask = [i for i in range(self.X.shape[1])
                if np.isnan(self.X[:, i]).mean() < null_count]

        self.X = self.X[:, mask]
        if self.features is not None:
            self.features = [self.features[i] for i in mask]

    def remove_low_variation(self, var_threshold=0):
        """function to remove features with little information as
        characterize by their variance"""

        mask = [i for i in range(self.X.shape[1])
                if self.X[:, i].var() > var_threshold]

        self.X = self.X[:, mask]
        if self.features is not None:
            self.features = [self.features[i] for i in mask]

    def impute_data(self, strategy='mean'):
        """Uses sklearn Imputer to impute null values with mean,
        median or most_frequent depending on the user input"""

        imp = SimpleImputer(strategy=strategy)
        self.X = imp.fit_transform(self.X)

    def scale_data(self, strategy='MinMaxScaler', **kwargs):
        """Uses skleran StandardScaler or MinMaxScaler to scale data"""

        if isinstance(strategy, str):
            strategy_mod = getattr(skpre, strategy)
        if not strategy == 'power_transform':
            scale = strategy_mod()
            self.X = scale.fit_transform(self.X)
            self.scale = scale
        else: 
            self.X = strategy_mod(self.X, **kwargs)

    def clean_data(self, scale_strategy=None):
        """Performs all the operations available."""
        self.remove_instance()
        self.remove_null_features()
        self.remove_low_variation()
        self.impute_data()
        if not scale_strategy:
            self.scale_data()
        else:
            self.scale_data(**scale_strategy)

    def get_PCA(se, n_pc=10):
        self.X_PCA = PCA(n_components=n_pc).fit_transform(self.X)

    def remove_highly_correlated_features():
        pass
