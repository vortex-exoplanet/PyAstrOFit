# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import numpy as np
import datetime
import os
import pickle
import emcee
import inspect

from astropy import constants as const
from astropy.time import Time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .Diagnostics import gelman_rubin
from .StatisticsMCMC import StatisticsMCMC
from .FileHandler import FileHandler
from . import Orbit
from .Planet_data import get_planet_data

from .Toolbox import cpu_count, now

__all__ = ['period',
           'semimajoraxis',
           'toSynthetic',
           'toKepler',
           'lnprior',
           'lnlike',
           'lnprob',
           "AISampler"]



packageName = 'PyAstrOFit'


# ------------------------
#   Top-level functions
# ------------------------

def period(semiMajorAxis=1, starMass=1):
    """
    """
    periodSecSquared = (np.power(semiMajorAxis * const.au.value,3) / (starMass * const.M_sun.value)) * 4 * np.power(np.pi,2) / const.G.value
    period = np.power(periodSecSquared,1./2.) / (365.*24*3600)
    return period


def semimajoraxis(period=1, starMass=1):
    """
    """
    aAUcubic = const.G.value * (starMass * const.M_sun.value) / (4 * np.power(np.pi,2)) * (period * 365.*24*3600)**2
    res_a = np.power(aAUcubic,1./3.) / const.au.value
    return res_a     


def toSynthetic(theta, which='Pueyo', mass=1, referenceTime=None):
    """
    """
    if which == 'Pueyo':
        res = np.zeros(6)
        temp = theta
        res[0] = math.log(period(temp[0],starMass = mass))
        res[2] = math.cos(math.radians(temp[2]))
        res[3] = np.mod(temp[4]+temp[3],360)
        res[4] = np.mod(temp[4]-temp[3],360)        
        res[1] = temp[1]
        res[5] = temp[5]
    elif which == 'alternative':
        res = np.zeros(6)
        temp = theta
        res[0] = math.log(period(temp[0],starMass = mass))
        res[2] = math.cos(math.radians(temp[2]))
        res[3] = temp[3]
        res[4] = temp[4]        
        res[1] = temp[1]
        res[5] = temp[5]        
    elif which == 'Chauvin':
        stat = StatisticsMCMC()
        res = stat.uFROMx(theta,referenceTime,mass)

    return res


def toKepler(u, which='Pueyo', mass=1, referenceTime=None):
    """
    """
    if which == 'Pueyo':
        res = np.zeros(6)
        res[1] = u[1]
        res[5] = u[5]
        
        res[0] = semimajoraxis(math.exp(u[0]), starMass = mass)
        res[2] = math.degrees(math.acos(u[2]))
        res[3] = np.mod((u[3]-u[4])*0.5,360)
        res[4] = np.mod((u[3]+u[4])*0.5,360)
        return res
    elif which == 'alternative':
        res = np.zeros(6)
        res[1] = u[1]
        res[5] = u[5]
        
        res[0] = semimajoraxis(math.exp(u[0]), starMass = mass)
        res[2] = math.degrees(math.acos(u[2]))
        res[3] = u[3]
        res[4] = u[4]
        return res        
    elif which == 'Chauvin':
        stat = StatisticsMCMC()
        res = stat.xFROMu(u,referenceTime,mass)    
        return res
    
    return None
    

def lnprior(theta, bounds=None, synthetic = False):
    """
    """
    u0, u1, u2, u3, u4, u5 = theta  
        
    if bounds is not None:
        if bounds[0][0] <= u0 <= bounds[0][1] and \
            bounds[1][0] <= u1 < bounds[1][1] and \
            bounds[2][0] <= u2 <= bounds[2][1] and \
            bounds[3][0] <= u3 <= bounds[3][1] and \
            bounds[4][0] <= u4 <= bounds[4][1]:                        
            return 0.0
        return -np.inf


def lnlike(theta, ob, er, priors, l, synthetic=False):
    """
    """
    if not synthetic:
        a, e, i, omega, w, tp = theta
    else:
        a, e, i, omega, w, tp = toKepler(theta, which=priors['which'], mass=priors['starMass'], referenceTime=priors['referenceTime'])
    
    orbit = Orbit(semiMajorAxis=a,eccentricity=e,inclinaison=i,longitudeAscendingNode=omega,periastron=w,periastronTime=tp,starMass=priors['starMass'], dStar=priors['starDistance'])
                  
    tano = np.array([orbit.trueAnomaly(ob['timePositionJD'][j]) for j in range(l)])                   
    t1 = [orbit.positionOnOrbit(trueAnomaly=tano[j],pointOfView="earth",unit="arcsec") for j in range(l)] 
    t2 = {'decPosition': np.array([t1[j]['decPosition'] for j in range(l)]),'raPosition': np.array([t1[j]['raPosition'] for j in range(l)]), 'timePosition': ob['timePosition']}
    model = np.concatenate((t2['raPosition'],t2['decPosition']))
    y = np.concatenate((ob['raPosition'],ob['decPosition']))
    yerr = np.concatenate((er['raError'],er['decError']))
    inv_sigma2 = 1.0/(yerr**2)
    
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

# ------------------------------------------------------------------------------------------------------------------------------------------ 
def lnprob(theta, ob, er, priors,l, synthetic):
    """
    """
    lp = lnprior(theta, bounds=priors['bounds'])
    if not np.isfinite(lp):
        return -np.inf    
    return lp + lnlike(theta, ob, er, priors, l, synthetic)



################################################################################################################################
################################################################################################################################
################################################################################################################################
class AISampler(FileHandler):
    """ 
    A sampler dedicated to orbit fitting using emcee package
    
    Parameters:
    -----------
    
    data:
        Path towards the data file.
        
    priors:
        Prior knowledge about the target, typically the star mass, the star
        distance, a prior vector of model parameters or model parameter bounds.
        Ex: priors = {'starMass' : 1.75, 'starDistance' : 19.3,
                      'bounds' : None, 'prior_model' : None}                                                                                  
    """
    
    def __init__(self,  data,
                        priors):
                            
        # TODO: autoriser l'utilisateur à ne pas faire de Gelman-Rubin                            
 
         # TODO: Revoir les méthodes Temporary et showCorner de AISampler
 
         # TODO: revoir la méthode showPDF

         # TODO: dans le notebook, avant de faire tourner le MCMC, on doit présenter les méthodes liés à la visualisation des data, ...
 
        # TODO: créer une méthode qui ajoute ou retire un point d'observation dans les data.                  

        # TODO: les highly probable initial guess devrait se trouver dans un fichier texte lui-même sauvegardé dans PyAstrOFit/rsc.
        
        """ The constructor of the class """
        # -----------------------------
        # Main part of the constructor
        # -----------------------------
         
        ## Get the data and convert the time of observation in JD                
        self._ob, self._er = get_planet_data(data)
        if self._ob is None and self._er is None:
            try:
                FileHandler.__init__(self,data,addContent=[])
                (self._ob, self._er) = FileHandler.getInfo(self)        
            except:
                raise Exception("Crash during the load of your data. Is the path correct ? --> {}".format(data))

        if len(list(self._ob.values())[0]) == 0:
            raise Exception("The file '{}' seems to be empty.".format(data))
        self._ob['timePositionJD'] = [Time(self._ob['timePosition'][k],format='iso',scale='utc').jd for k in range(len(self._ob['timePosition']))]
        self._l = len(self._ob['timePositionJD'])                
                   
        self._data = data                 
        self._ndim = 6     
        self._pKey = ['semiMajorAxis','eccentricity','inclinaison', 'longitudeAscendingNode', 'periastron', 'periastronTime']    

        ## Check what contains the attribut priors and initialize the initial state self._theta0
        self._priors, self._theta0 = AISampler.priors_check(self,priors)
        
        ## List class attributs
        self._list = [s for s in dir(self) if s[0] == '_' and s[1] != '_' and s[1:12] != 'FileHandler' and s[1:4] != 'get']   

    # ------------------------------------------------------------------------------------------------------------------------------------------            
    # ---------
    # Property
    # --------     
            
    # --------- PRIORS            
    @property
    def priors(self):
        return self._priors
        
    @priors.setter
    def priors(self,new_value):
        new_value, self._theta0 = AISampler.priors_check(self,new_value)
        self._priors = new_value        

    # --------- THETA0    
    @property
    def firstguess(self):          
        return self._theta0

    # --------- OBSERVATIONS    
    @property
    def data(self):          
        return self._ob 
        
    # --------- ERRORS    
    @property
    def errors(self):          
        return self._er         
        
        
    # --------
    # Methods
    # --------    
    # ------------------------------------------------------------------------------------------------------------------------------------------        
    def class_attribut(self, **kwargs):
        """
        Return a dict with the class attributs. Additional attribut can be added.
        
        """                
        temp = {key[1:]: vars(self)[key] for key in self._list}                
                
        for key in kwargs:
            temp[key] = kwargs[key]

        return temp  
                
    # ------------------------------------------------------------------------------------------------------------------------------------------        
    def showData(self, prior_model = False, lim = None, figsize = (8,8)):
        """
        Display a figure wich represents the data with their error bars.
        
        Parameters:
        -----------
        prior_model: boolean
            If true, the orbit corresponding to the prior model is represented.
            
        lim: list
            x and y axis bounds. Ex: lim = [[0.2,0.5],[-0.3,-0.1]]
        
        figsize: tuple
            Size of the returned figure. Ex: figsize = (6,6).
            
        """
        if self.priors['prior_model'] is None or not prior_model:
            orbit = Orbit()
            show = False
        else:
            orbit = Orbit(semiMajorAxis=self.priors['prior_model'][0],
                            eccentricity=self.priors['prior_model'][1], 
                            inclinaison=self.priors['prior_model'][2],
                            longitudeAscendingNode=self.priors['prior_model'][3],
                            periastron=self.priors['prior_model'][4],
                            periastronTime=self.priors['prior_model'][5],
                            dStar=self.priors['starDistance'],
                            starMass=self.priors['starMass'])
            show = True                            
        
        if lim is None:
            ra_min = np.array(self.data['raPosition']).min()
            ra_max = np.array(self.data['raPosition']).max()
            dec_min = np.array(self.data['decPosition']).min()
            dec_max = np.array(self.data['decPosition']).max()
            index = 0.1
            lim = [[ra_min*(1-index*np.sign(ra_min)),ra_max*(1+index*np.sign(ra_max))],[dec_min*(1-index*np.sign(dec_min)),dec_max*(1+index*np.sign(dec_max))]] 
                    
        orbit.showOrbit(figsize=figsize,
                        unit='arcsec',
                        addPoints=[self.data['raPosition'],self.data['decPosition'],self.errors['raError'],self.errors['decError']],
                        lim=lim,
                        show = show)
              
    # ------------------------------------------------------------------------------------------------------------------------------------------        
    def run_mcmc(self,
                 nwalkers,
                 a = 2.0,
                 pos_coeff = np.array([1e-02,1e-02,1e-02,1e-02,1e-02,1e-05]),
                 burnin = 0.5, 
                 itermin = None,
                 limit = None, 
                 supp = 0, 
                 fraction = 0.5,
                 maxgap = 1e+04,
                 grThreshold = np.ones(6)*1.01, 
                 grCountThreshold = 3, 
                 synthetic = False,
                 threads = 1,
                 output = None,                 
                 verbose = True, 
                 showWalk = False, 
                 showCorner = False, 
                 temporary = True):
        """
        Run an affine invariant mcmc algorithm in order to draw marginalized 
        distribution for the 6 Kepler orbital parameters.
        
        Parameters
        ----------
        nwalkers: int
            The number of Goodman & Wear "walkers".
        
        a: float
            The proposal scale parameter. (default: "2.0")
            
        burnin:
            The fraction of a walker which is discarded. (default: "0.0")  

        itermin:
            Steps per walker lower bound. The simulation will run at least this
            number of steps per walker. (default: "1e03")             
        
        limit:
            Steps per walker upper bound. If the simulation runs up to ''limit''
            steps without having reached the convergence criterion, the run
            is stopped. (default: "None")
        
        supp:
            Number of extra steps per walker after having reached the convergence
            criterion. (default: 0.0)
            
        maxgap: int, optional
            Maximum number of steps per walker between two Gelman-Rubin test.
        
        grThreshold:
            The Gelman-Rubin threshold used for the test for nonconvergence. 
            (default: "1.01")
            
        synthetic: boolean
            If True, the vector parameters to be fitted will be 
            [log(P), e, cos(i), Omega, omega, t_omega] instead of the Kepler
            parameters. 
            
        threads: int, optional
            The number of threads to use for parallelization. (default: "1")            
        
        output: str, optional
            The path which roots to the stored result files.  
            
        verbose: boolean, optional
        
        showWalk: boolean, optional
            If True, the current walk plot is displaying at each statistical test.
            
        showCorner: boolean, optional
            If True, the current corner plot is displaying at each statistical
            tests.
        temporary: boolean, optional
            If True, the chain and lnprobability are stored in the sampler_emcee
            object. I will increase the size of the output file. So, prefer False
            only if you need lnprobabiliy being recovered.
            
        Returns
        -------
        out : AISamplerResults instance.    
        
        Notes
        -----
        The parameter 'a' must be > 1.
        
        The parameter 'rhat_threshold' can be a numpy.array with individual 
        threshold value for each model parameter.        
        
        """

        # Let's go !        
        start_time = now() 
        if verbose:
            print('')
            print('################################################################')
            print('The MCMC run has started.')
            print('')
            print('Start time: {}:{}:{}'.format(start_time[0],start_time[1],start_time[2]))
            print('################################################################')


        # Output
        if output is not None:
            if not os.path.exists(output):
                os.makedirs(output)


        # Synthetic ?
        if synthetic:
            self._theta0 = toSynthetic(self._theta0, which = self._priors['which'], mass = self._priors['starMass'], referenceTime = self._priors['referenceTime'])
            bounds_lower_new = toSynthetic(self._priors['bounds'][:,0], which=self._priors['which'],mass=self._priors['starMass'])
            bounds_upper_new = toSynthetic(self._priors['bounds'][:,1], which=self._priors['which'],mass=self._priors['starMass'])
            self._priors['bounds'] = np.array([np.sort([bounds_lower_new[j],bounds_upper_new[j]]) for j in range(self._ndim)])

                        
        # Initialization
        chain = np.zeros([nwalkers,1,self._ndim]) 
            
        if threads == -1:
            threads = cpu_count()   
        else:
            threads = threads
                  
        sampler_emcee = emcee.EnsembleSampler(nwalkers,self._ndim,lnprob, a ,args=(self._ob,self._er,self._priors,self._l,synthetic),threads=threads)
                
        if burnin <= 0 or burnin >= 1:
            burnin = 0.5
            print('The burnin parameter should be 0 < burnin < 1. The value {} has been automatically modified to 0.5'.format(burnin))
        
        if itermin is None:
            itermin = 2e+04/nwalkers
                    
        if limit is None:
            limit = 1e05/nwalkers
        elif limit == -1:
            limit = 5e08/nwalkers            
        
        iterations = limit + supp        
        lastcheck = 0
        konvergence = np.inf
        stop = np.inf
        grCount = 0
        rhat = np.zeros(self._ndim)
        geom = 0
        
        # Define the initial state for walkers 
        if self._theta0 is not None:                
            temp0 = np.random.normal(self._theta0[0],self._theta0[0]*pos_coeff[0],nwalkers)
            temp1 = np.random.normal(self._theta0[1],self._theta0[1]*pos_coeff[1],nwalkers)
            temp2 = np.random.normal(self._theta0[2],self._theta0[2]*pos_coeff[2],nwalkers)
            temp3 = np.random.normal(self._theta0[3],self._theta0[3]*pos_coeff[3],nwalkers)
            temp4 = np.random.normal(self._theta0[4],self._theta0[4]*pos_coeff[4],nwalkers)
            temp5 = np.random.normal(self._theta0[5],self._theta0[5]*pos_coeff[5],nwalkers)            
            self._pos = np.transpose(np.vstack((temp0,temp1,temp2,temp3,temp4,temp5)))            
        else:
            self._pos = np.random.rand(nwalkers,self._ndim)
            for j in range(self._ndim):
                self._pos[:,j] = self._pos[:,j]*np.diff(self._priors['bounds'])[j][0] + self._priors['bounds'][j][0]
            
        # Start the chain construction
        if verbose:
            print('The construction of {} walkers has started ...'.format(nwalkers))
            print('')            
        start = datetime.datetime.now()
        
        for k, res in enumerate(sampler_emcee.sample(self._pos,
                                                     iterations=iterations,
                                                     storechain=temporary)):
            chain = AISampler.storechain(self, k, res[0], chain)

            ## Criterion for statistical test
            if k == np.amin([math.ceil(itermin*(1+fraction)**geom),lastcheck+math.floor(maxgap)]):
                geom += 1
                lastcheck = k
                if verbose:
                    print("--> {} steps per walker".format(k))
                
                ar = np.median(sampler_emcee.acceptance_fraction)
                if verbose:
                    print("   Median acceptance rate (AR) {}".format(ar))
                if ar < 0.2 or ar > 0.5:
                    if verbose:
                        print('   >> AR is out of the limit. The hyperparameter a ({}) should be adjusted.'.format(a))
                                    
                if showWalk:
                    if not synthetic:
                        AISampler.showWalk(self, isamples = AISampler.chain_zero_truncated(self,chain), burnin = 0)
                    else:
                        if self._priors['which'] == 'Pueyo':
                            labels = ["$\log(P)$","$e$","$\cos(i)$","$\omega+\Omega$ (deg)","$\omega-\Omega$ (deg)","$t_\omega$ (JD)"]
                        elif self._priors['which'] == 'alternative':
                            labels = ["$\log(P)$","$e$","$\cos(i)$","$\Omega$ (deg)","$\omega$ (deg)","$t_\omega$ (JD)"]                            
                        elif self._priors['which'] == 'Chauvin':
                            labels = ["$u_1$","$u_2$","$u_3$","$u_4$","$u_5$","$u_6$"]
                        AISampler.showWalk(self, isamples = AISampler.chain_zero_truncated(self,chain), burnin = 0, labels=labels)
                
                if showCorner:
                    lisa = chain.shape[1]
                    chain_zero_truncated = AISampler.chain_zero_truncated(self,chain)
                    AISampler.showPDFCorner(self,isamples = chain_zero_truncated[:,int(np.floor(lisa*burnin)):,:].reshape((-1,6)), labels=labels)
                                        
#                if temporary: # under construction                   
#                    #isamples_pickle = AISampler.independentSamples(self, chain, length='burnin', burnin=burnin)
#                    #AISampler.save_pickle(self)
#                    pass

                
                if k >= limit-1:
                    if verbose:
                        print("We have reached the limit number of steps without having converged")
                    break
                elif (k+1) >= itermin and konvergence == np.inf:
                    threshold = int(np.floor(burnin*k))
                    threshold2 = int(np.floor((1-burnin)*k*0.25))
                    for j in range(self._ndim):
                        part1 = chain[:,threshold:threshold+threshold2,j].reshape((-1))
                        part2 = chain[:,threshold+3*threshold2:threshold+4*threshold2,j].reshape((-1))
                        series = np.vstack((part1,part2))
                        #rhat[j] = AISampler.gelman_rubin(self,series)
                        rhat[j] = gelman_rubin(series)
                    
                    rhat_trunc = np.trunc(rhat*1000)/1000
                    if verbose:
                        print("   Gelman-Rubin hat_R = {}".format(rhat_trunc))
                        print("")                    
                    if (rhat <= grThreshold).all():
                        grCount += 1
                        if verbose:
                            print("      Gelman-Rubin test OK {}/{}".format(grCount,grCountThreshold))
                        if grCount >= grCountThreshold:
                            #finish_konvergence = datetime.datetime.now()
                            #self._elapsedTimeKonvergence = finish_konvergence-start
                            # we have to run 1e06/nwalerks steps more
                            if verbose:
                                print("      All the parameters have passed the Gelman-Rubin test (threshold = {}): {}".format(grThreshold,rhat))
                            konvergence = k
                            stop = konvergence + supp
                            if verbose:
                                print("")
                                print("   Construction of isamples")
                    else:
                        grCount = 0
                    
            
            if (k+1) >= stop:
                if verbose:
                    print("We have finished to run {} steps more (per walker) after the convergence".format(supp))
                break
                 
        
        if k >= limit-1 and verbose:
                print('We have reached the maximum number of steps per walker ({}).'.format(limit))
 
        finish = datetime.datetime.now()   
        elapsedTime = finish-start

        # Dict of input parameters with their associated values
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        input_parameters = {j : values[j] for j in args[1:]}
        
        # Dict of useful internal "run_mcmc" parameters with their associated values
        internal_parameters = {'konvergence' : konvergence, 
                               'duration' : elapsedTime,
                               'date_of_run' : datetime.datetime.now()}
        
        if verbose:
            end_time = now()
            print('')
            print('################################################################')
            print('The MCMC run has finished.')
            print('')
            print('End time: {}:{}:{}'.format(end_time[0],end_time[1],end_time[2]))
            print('Total duration: {}'.format(elapsedTime))
            print('################################################################')


        return AISamplerResults(AISampler.chain_zero_truncated(self,chain),
                                input_parameters,
                                internal_parameters,
                                sampler_emcee,
                                AISampler.class_attribut(self))

    # ------------------------------------------------------------------------------------------------------------------------------------------        
    def storechain(self, k, walker_current_position, chain):
        """
        Returns the chain updated with the new position.
        
        """
        # check if the size of chain is correct. If not, increase the size of the chain
        s = chain.shape[1]
        nwalkers = chain.shape[0]
        if k+1 > s:
            empty = np.zeros([nwalkers,2*s,self._ndim])
            chain = np.concatenate((chain,empty),axis=1)
        
        # store the state of the chain
        chain[:,k] = walker_current_position
        return chain



        
    # ------------------------------------------------------------------------------------------------------------------------------------------ 
    def chain_zero_truncated(self,chain): 
        """
        Return the Markov chain with the dimension: walkers x steps* x parameters,
        where steps* is the last step before having 0 (not yet constructed chain)
        
        """
        try:
            idxzero = np.where(chain[0,:,0] == 0.0)[0][0]
        except:
            idxzero = chain.shape[1]        

        return chain[:,0:idxzero,:]                
            
        
    # ------------------------------------------------------------------------------------------------------------------------------------------                            
#    def save_txt(self, chain = None, output = None):   
#        """Save the Markov chain in a txt file."""        
#        # OUTPUT NAME/PATH FILE
#        if output == None:
#            output = self._outputpath + 'chain.txt'
#        
#        # CHAIN
#        if chain is None:
#            chain = self.chain.reshape((-1,6))
#        else:
#            if len(chain.shape) != 2:
#                chain = chain.reshape((-1,6))
#        
#        # FILE
#        #if 100./(6*13/1e+06) > chain.shape[0]:
#        #    print 'File size > 100Mo' 
#            
#        with open(output,"a") as f:
#            for line in chain:
#                f.write(''.join(['\t\t'.join(map(str,line)),'\n']))            
#        
#        print('The file '+ output +' has been successfully saved.') 

                

#    # ------------------------------------------------------------------------------------------------------------------------------------------
#    def showPDF(self, parameters = range(6), isamples = None, burnin = 0.5, nBin = 40, save = False, output = ''):
#        xHist, bins = dict(), dict()
#        #nBin = 40
#                            
#        if isamples == None:
#            if self._isamples.size == 0:
#                idx0 = np.where(self._chain[0,:,0] == 0)[0][0]
#                chain = self._chain[:,int(np.floor(burnin*(idx0-1))):idx0,:].reshape((-1,self._ndim))
#            else:
#                chain = self._isamples
#        else:
#            if len(isamples.shape) > 2 and isamples.shape[2] == 6: 
#                chain = isamples.reshape((-1,6))
#            else:
#                chain = isamples
#        
#        labels = ["$a$ (AU)","$e$","$i$ (deg)","$\Omega$ (deg)","$\omega$ (deg)","$t_\omega$ (JD)"]
#        for k in parameters:
#            fig = plt.figure()
#            plt.hold('on')
#            xHist[self._pKey[k]], bins[self._pKey[k]] = np.histogram(chain[:,k],bins = nBin,density=False)
# 
#            xaxis = (bins[self._pKey[k]][:-1] + bins[self._pKey[k]][1:])/2
#            yaxis = xHist[self._pKey[k]]       
#
#            plt.plot(xaxis,yaxis,'k', alpha=0.8)
#            #plt.plot(self._theta0[k]*np.ones(2),[0,np.amax(xHist[self._pKey[k]])],'k--',alpha=0.5)
#            plt.xlabel(labels[k], fontsize=12)
#            plt.ylabel('$N_{Orbits}$', fontsize=12)
#            plt.suptitle('POSTERIOR DISTRIBUTION  - '+labels[k]+' -', fontsize=12, fontweight='bold')
#            #plt.set_title(whichSampler[0], fontsize=12, fontweight='bold')
#            #plt.set_xlim([xlimAll[str(whichMC[0])][pKey[k]][0][0]-180*peri,xlimAll[str(whichMC[0])][pKey[k]][0][1]-180*peri])
#            #plt.set_ylim([0,xlimAll_fig['0'][pKey[k]][1][1]/8.])
#            if save:
#                plt.savefig(output+'pdf_'+str(k)+'.pdf')
#                plt.close(fig)
#            else:
#                plt.show()


    # ------------------------------------------------------------------------------------------------------------------------------------------        
    def showPDFCorner(self, isamples, burnin = 0.5, labels = None, save = False, output = ''):
        """
        Display or save the so-called corner plot which shows (when the 
        convergence is reached) the marginalized probability distributions and 
        the correponding correlation between various parameters.
        
        Parameters:
        -----------
        isamples: np.array (optional)
        
        burnin: float (optional)
        
        save: boolean (optional)        
        """
        
        # DEPENDENCY
        try:
            import corner as triangle
        except:
            raise Exception("You should install the Triangle package: https://github.com/dfm/corner.py")

        # ISAMPLES
        if isamples is None:
             
            length = self.chain.shape[1] 
            isamples = self.chain[:,int(np.floor(burnin*length)):-1,:].reshape((-1,self._ndim))    

        else:
            if len(isamples.shape) > 2 and isamples.shape[2] == 6: 
                length = isamples.shape[1]
                isamples = isamples[:,int(np.floor(burnin*length)):-1,:].reshape((-1,6))

        # LABELS        
        if labels is None:        
            labels = ["$a$ (AU)","$e$","$i$ (deg)","$\Omega$ (deg)","$\omega$ (deg)","$t_\omega$ (JD)"]       
            
        # FIGURE    
        if isamples.shape[0] == 0:
            print("It seems that the chain is empty. Have you already run the MCMC ?")
        else:
            #labels = ["$a$ (AU)","$e$","$i$ (deg)","$\Omega$ (deg)","$\omega$ (deg)","$t_\omega$ (JD)"]
            fig = triangle.corner(isamples, labels=labels)    
        
        # SAVE
        if save:
            plt.savefig(output+'corner_plot.pdf')
            plt.close(fig)
        else:
            plt.show()

    # ------------------------------------------------------------------------------------------------------------------------------------------                
    def showWalk(self, isamples, burnin = None,  figsize = (8,8), labels = None, save = False, output = ''):
        """
        Display or save the so-called walk plot wich show the construction progress for each walker.
        
        Parameters:
        -----------
        isamples: np.array (optional)
        
        burnin: float (optional)
        
        figsize: tuple (optional)
        
        labels: list (optional)
        
        save: boolean (optional)
        
        """
        # ISAMPLES
        #if isamples is None:
        #    isamples = self.chain
                
        # LABELS        
        if labels is None:        
            labels = ["$a$ (AU)","$e$","$i$ (deg)","$\Omega$ (deg)","$\omega$ (deg)","$t_\omega$ (JD)"]
        else:
            labels = labels

        # FIGURE
        fig, axes = plt.subplots(6, 1, sharex=True, figsize=figsize)
        for j in range(6):            
            axes[j].plot(isamples[:,:,j].T, color="k", alpha=0.4)
            axes[j].yaxis.set_major_locator(MaxNLocator(5))
            axes[j].set_ylabel(labels[j])
            axes[j].set_xlim([0,isamples.shape[1]])
        axes[j].set_xlabel("step number")        
        fig.tight_layout(h_pad=0.0)
        
        # SAVE
        if save:
            plt.savefig(output+'walk_plot.pdf')
            plt.close(fig)
            print('The file {} has been successfully saved.'.format(output+'walk_plot.pdf'))
            
        else:
            plt.show()


    # ------------------------------------------------------------------------------------------------------------------------------------------
    def priorSemiMajorAxis(self):
        """
        Returns a low bound for the semi-major axis, according to the data and star priors.
        
        """
        return np.amax((self._ob['decPosition']**2 + self._ob['raPosition']**2)**(0.5))/3600./180.*np.pi*(self._priors['starDistance']*3.08567758e+16/1.49597871e+11)
        
        
    # ------------------------------------------------------------------------------------------------------------------------------------------
    def priors_check(self,priors):
        """
        Check what contains the attribut priors. Modify it if necessary or raise an exception.
        
        """
        if not isinstance(priors,dict):
            raise Exception('The attribut "priors" must be a dictionary, {} has be given'.format(type(priors)))
        
        mandatory_keys = ['starMass','starDistance']
        if not all(key in priors.keys() for key in mandatory_keys):
            raise Exception('''The attribut "priors" is a dictionary which must contains at least the following keys: starMass, starDistance''')
            

        if 'bounds' not in priors.keys():
            priors['bounds'] = np.array([[0.1,200],[0,1],[0,180],[0,360],[0,360],[2457190-365*500,2457190+365*500]]) 
        elif priors['bounds'] is None: 
            priors['bounds'] = np.array([[0.1,200],[0,1],[0,180],[0,360],[0,360],[2457190-365*500,2457190+365*500]]) 
            

        if 'prior_model' not in priors.keys():
            priors['prior_model'] = None                                 
            

        if 'which' not in priors.keys():
            priors['which'] = 'alternative'  
            

        if 'referenceTime' not in priors.keys():
            priors['referenceTime'] = None   
            
        theta0 = np.zeros(6)    
        if priors['prior_model'] is None: # TODO: ces valeurs devraient se trouvées dans un fichier txt.
            if self._data == 'betapicb':                                                                        
                theta0 = np.array([8.42,0.07,89.04,31.87,338.25,2452353.32])
            elif self._data == 'hr8799b':
                theta0 = np.array([67.95,0.001,22.06,55.5,0.01,2451259.5])
            elif self._data == 'hr8799c':
                theta0 = np.array([42.81,0.001,29.18,63.65,0.01,2395666.125])
            elif self._data == 'hr8799d':
                theta0 = np.array([26.97,0.001,37.28,57.90,0.01,2439013.3333333335])
            elif self._data == 'hr8799e':
                theta0 = np.array([14.81,0.001,21.38,83.93,0.01,2448045.7916666665])                
            else:
                theta0 = None                
            priors['prior_model'] = theta0    
        else:
            theta0 = priors['prior_model'] 
 
        return priors, theta0   


################################################################################################################################
################################################################################################################################
################################################################################################################################

class AISamplerResults(AISampler):
    """
    Define a specific object for the MCMC results.
    
    """
    def __init__(self, chain,
                       input_parameters,
                       internal_parameters,
                       sampler_emcee,
                       sampler_parameters):
        
        self._chain = chain
        self._input_parameters = input_parameters
        self._internal_parameters = internal_parameters
        self._sampler_parameters = sampler_parameters                 

        self._lnprobability = sampler_emcee.lnprobability
        sampler_emcee.reset()
        self._sampler_emcee = sampler_emcee
        
        
                       

    # ---------
    # Property
    # ---------
    # --------- CHAIN and LNPROBABILITY   
    @property
    def chain(self):
        return self._chain

    @property
    def lnprobability(self):
        return self._lnprobability        
                              
    # --------- INPUT PARAMETER
    @property
    def input_parameters(self):
        return self._input_parameters
        
    # --------- INTERNAL PARAMETER
    @property
    def internal_parameters(self):
        return self._internal_parameters
                      
    # --------- EMCEE SAMPLER
    @property
    def sampler_emcee(self):
        return self._sampler_emcee  
        
    # --------- SAMPLER PARAMETER
    @property
    def sampler_parameters(self):
        return self._sampler_parameters        

    # --------
    # Methods
    # --------
    # ------------------------------------------------------------------------------------------------------------------------------------------                            
    def save_pickle(self, output = None, isamples = False, length = None):
        """Pickle the Markov chain in a file""" 
        if output is None:
            output = self.input_parameters['output'] + 'run_results'
            if not os.path.exists(self.input_parameters['output']):
                os.makedirs(self.input_parameters['output'])
        else:
            index = output.rfind('/')            
            if not os.path.exists(output[:index+1]) and index != -1:
                os.makedirs(output[:index+1])
            
            
        if isamples:
            chain = AISamplerResults.independentSamples(self, length = length)
        else:
            chain = self.chain
                
            
        results = {'input_parameters' : self.input_parameters, 
                   'internal_parameters' : self.internal_parameters, 
                   'sampler_parameters' : self.sampler_parameters,
                   'chain' : chain,
                   'lnprobability': self._lnprobability}

        with open(output,'wb') as fileSave:
            myPickler = pickle.Pickler(fileSave)
            myPickler.dump(results)         
      
        print('The file has been successfully saved:\n{}'.format(os.getcwd()+'/'+output))        


    # ------------------------------------------------------------------------------------------------------------------------------------------ 
    def showWalk(self, isamples = None, burnin = None,  figsize = (8,8), labels = None, save = False, output = None):
        """
        Override the AISampler showWalk method.
        
        """
        if self.input_parameters['synthetic']: #TODO : il faut également regarder quel type de synthetic
            if self.sampler_parameters['priors']['which'] == 'alternative':
                labels = ["$\log(P)$","$e$","$\cos(i)$","$\Omega$ (deg)","$\omega$ (deg)","$t_\omega$ (JD)"]
        
        if output is None:
            output = self.input_parameters['output']
         
        if isamples is None:
            return super(AISamplerResults, self).showWalk(self._chain, burnin,  figsize, labels, save, output)
        else:
            return super(AISamplerResults, self).showWalk(isamples, burnin,  figsize, labels, save, output)
        

    # ------------------------------------------------------------------------------------------------------------------------------------------        
    def showPDFCorner(self, isamples = None, burnin = 0.5, labels = None, save = False, output = None):  
        """
        Override the AISampler showCorner method.
        
        """
        if self.input_parameters['synthetic']: #TODO : il faut également regarder quel type de synthetic
            if self.sampler_parameters['priors']['which'] == 'alternative':
                labels = ["$\log(P)$","$e$","$\cos(i)$","$\Omega$ (deg)","$\omega$ (deg)","$t_\omega$ (JD)"]
        
        if output is None:
            output = self.input_parameters['output']    
        
        if isamples is None:
            return super(AISamplerResults, self).showPDFCorner(self._chain, burnin, labels, save, output)
        else:
            return super(AISamplerResults, self).showPDFCorner(isamples, burnin,labels, save, output)


    # ------------------------------------------------------------------------------------------------------------------------------------------        
    def independentSamples(self, chain = None, length = None, burnin = None):
        """ Extract independent samples from the chain. 
        
        If the convergence is not reached, the isamples will be a copy of the 
        current state of the chain.
        
        :param length:
            The number of independent samples randomly extracted from the chain.
            
        """        
        if burnin is None:
            burnin = self._input_parameters['burnin']
        
        if chain is None:
            chain = self.chain        
      
        isamples_full = chain[:,np.floor(burnin*chain.shape[1]):,:].reshape((-1,self.sampler_parameters['ndim']))
        size = isamples_full.shape[0]

        if length is None:
            return isamples_full
        else:
            if length > size:
                print('The parameter length ({}) must be <= the length of the Markov chain ({}).'.format(length,size))
                length = size
            return isamples_full[np.random.choice(range(size),size=length,replace=False),:]            
            
           