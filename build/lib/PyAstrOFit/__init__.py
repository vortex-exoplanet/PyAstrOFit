# -*- coding: utf-8 -*-
"""
 MoucMouc v 1.0
 
"""

#from .Orbit import *
#from .Toolbox import *
#from .FileHandler import *
#from .StatisticsMCMC import *
from .Sampler import *
from .BayesianInference import *

__version__ = "1.0.0"


# TODO: checker aussi les packages comme psutil, ... mais simplement dire (par un warning) que s'il n'est pas installé
# certaines méthodes ne fonctionneront pas.

print("Check the dependencies ...")
try:
    import numpy
    print("+ The numpy package is installed.")
except ImportError:
    raise("+ The package -Numpy- is required to go forward. Please install this package.")   

try:
    import matplotlib.pyplot as plt
    print("+ The matplotlib package is installed.")
except ImportError:
    raise("+ The package -matplotlib- is required to go forward. Please install this package.")

try:
    import astropy
    print("+ The astropy package is installed.")
except ImportError:
    raise("+ The package -astropy- is required to go forward. Please install this package.")  

try:
    import emcee
    print("+ The emcee package is installed.")
except ImportError:
    raise("+ The package -emcee- is required to go forward. Please install this package.")  
 
#try:
#    import PyAstronomy
#    print("+ The PyAstronomy package is installed.")
#except ImportError:
#    raise("+ The package -PyAstronomy- is required to go forward. Please install this package.")   
 

 
