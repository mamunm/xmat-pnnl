from .utilities import load_data
from .poly_fit import PolyFit
from .shapley_data import Shapley
from .constant_matching import ConstantMatcher
from .linear_model_zoo import LinearModelZoo 
from .sklearn_gp_wrapper import SKGP
from .data_processing import ProcessData
from .shapley_features import ShapleyFeatures

__all__ = ['load_data', 
           'PolyFit', 
           'Shapley', 
           'ConstantMatcher',
           'LinearModelZoo',
           'SKGP',
           'ProcessData',
           'ShapleyFeatures']
__version__ = '0.1'
