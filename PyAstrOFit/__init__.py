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
#from .Planet_data import *

__version__ = "1.0.0"


print "________                      _                                  ____    ________"
print "`MMMMMMMb.                   dM.                                6MMMMb   `MMMMMMM 68b"         
print " MM    `Mb                  ,MMb                /              8P    Y8   MM    \ Y89   /"     
print " MM     MM ____    ___      d'YM.      ____    /M     ___  __ 6M      Mb  MM      ___  /M"     
print " MM     MM `MM(    )M'     ,P `Mb     6MMMMb\ /MMMMM  `MM 6MM MM      MM  MM   ,  `MM /MMMMM"  
print " MM    .M9  `Mb    d'      d'  YM.   MM'    `  MM      MM69   MM      MM  MMMMMM   MM  MM"     
print " MMMMMMM9'   YM.  ,P      ,P   `Mb   YM.       MM      MM'    MM      MM  MM   `   MM  MM"     
print " MM           MM  M       d'    YM.   YMMMMb   MM      MM     MM      MM  MM       MM  MM"     
print " MM           `Mbd'      ,MMMMMMMMb       `Mb  MM      MM     YM      M9  MM       MM  MM"     
print " MM            YMP       d'      YM. L    ,MM  YM.  ,  MM      8b    d8   MM       MM  YM.  ," 
print "_MM_            M      _dM_     _dMM_MYMMMM9    YMMM9 _MM_      YMMMM9   _MM_     _MM_  YMMM9" 
print "               d"                                                                             
print "           (8),P"                                                                              
print "            YMM"

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
 

 
