#!/usr/bin/env python
 
import os
#import sys
#import re

try:
    from setuptools import setup, find_packages
    setup
except ImportError:
    from distutils.core import setup
    setup
 

readme = open('readme.rst').read()
doclink = """
Documentation
-------------
There are IPython notebook (*.ipynb files) dedicated to explain how to use this 
package.  
"""
 
PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))
 
setup(
    name='PyAstrOFit',
    version='0.0a1',
    description='MCMC dedicated to planet orbital fit.',
    long_description=readme + '\n\n' + doclink + '\n\n',
    author='Olivier Wertz',
    author_email='owertz@ulg.ac.be',
    url='https://github.com/owertz/PyAstrOFit.git',                    
    packages=find_packages(),
    include_package_data=True,
    install_requires=['numpy >= 1.8.1',
                      'ipython',
                      'astropy >= 1.0.2',
                      'matplotlib',
                      'emcee'
                      ],
    license=' not defined ',
    zip_safe=False,
    keywords='PyAstrOFit',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        "Intended Audience :: Science/Research",
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ] 
)