
 PyAstrOFit
------------------------------------
  Python Astrometric Orbit Fitting  
------------------------------------

PyAstrOFit is a package dedicated to orbital fit within the
MCMC approach, written in Python 2.7. The MCMC sampler makes
direct use of emcee (Foreman-Mackey et al. 2013).

This package works and is stable. However, it still requires substantial 
improvement. However, you can find at the root three IPython notebooks (*.ipynb files) 
dedicated to explain how to use this package. 

PyAstrOFit is being developed within the VORTEX team @ University of Liege (Belgium).
It's still a work in progress. If you want to report a bug, suggest a feature or add a 
feature please contact the main developer at owertz [at] ulg.ac.be or through 
github.


QUICK INSTRUCTIONS
==================
From the root of the PyAstrOFit package:
$ python setup.py install

Or if you want to keep the trace off all installed file:
$ python setup.py install —-record files.txt

Therefore, if you want to uninstall the package, just do:
$ cat files.txt | xargs rm -rf


DEPENDENCIES
============
You must have a python distribution installed (e.g. Canopy, Anaconda, MacPorts),
that will allow easy and robust package management and avoid messing up with the 
system default python. I recommend using Anaconda over Canopy or MacPorts. 

The PyAstrOFit package depends on existing packages from the Standard Library
and specific ones as well, e.g.:

numpy
matplotlib
astropy
