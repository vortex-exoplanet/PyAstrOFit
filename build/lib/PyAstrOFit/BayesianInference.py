# -*- coding: utf-8 -*-


import numpy as np
import pickle

from astropy.time import Time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

#from StatisticsMCMC import StatisticsMCMC as st
from FileHandler import FileHandler as fh
import Orbit as o
from Planet_data import get_planet_data
from Sampler import toKepler

try:
    import triangle
except ImportError:
    raise Exception("You should install the Triangle package: https://github.com/dfm/triangle.py")
    
    
__all__ = ["BayesianInference"]


#packageName = 'PyAstrOFit'


class BayesianInference(object):
    """ 
    Class. 
    Tools to perform Bayesian Inference and Markov chains representation.
    
    Parameters
    ----------
    markovchain: array-like
        The Markov chain obtained from PyAstrOFit.AISampler object. The shape 
        is (nwalker, steps, dim), where -nwalker- represents the number of walkers, 
        -steps- the number of steps per walker and -dim- the number of model
        parameters.
    
    data: str
        The user file path containing positions and errors at different epochs;
        or a keyword which is passed to PyAstrOFit.Planet_data.get_planet_data().
        For instance: data = 'hr8799b'.
         
    priors: dict
        Dictionary which must contains -starDistance- and -starMass- keys. Other
        can be given as well: -bounds-, -priors_model-, -which- or 
        -referenceTime-

    lnprobability: array-like (optional)
        (walker x steps) array which contains the log-likelihood of each step 
        in the Markov chain (markovchain).
    
    outputPath: str
        The results file path.
          
    Methods
    -------
    independent_samples
    reset
    showPDFCorner
    showWalk
    confidence  
    best_solution
    solution_in_confidence    
    highlyProbableSolution  
    showSolutions           
               
    """
    
    # -----------------------
    # Class attributs
    # -----------------------        
    
    
    def __init__(self,  markovchain,
                        data,
                        priors,
                        lnprobability=None,
                        input_parameters=None,
                        outputPath=''):
                            
        """ The constructor of the class """
        ### TODO: les attributs qui ne nécessitent pas d'appel via @property 
        ###       devraient être renommé self.attribut au lieu de self._attribut
        
        # Priors and Input parameters
        self._priors = priors
        self._input_parameters = input_parameters 
        
        if input_parameters is not None:
            self._synthetic = self._input_parameters['synthetic']
        else:
            self._synthetic = False
       
       
        # Markov chain
        if self._synthetic:
            self.markovchain_synthetic = markovchain
            self.markovchain = np.zeros_like(markovchain)            
            for j, walker in enumerate(self.markovchain_synthetic):
                self.markovchain[j,:,:] = np.array([toKepler(parameters,
                                                             which=self._priors['which'],
                                                             mass=self._priors['starMass'],
                                                             referenceTime=self._priors['referenceTime']) for parameters in walker])            
        else:
            self.markovchain = markovchain
            self.markovchain_synthetic = np.zeros_like(markovchain)
            

        self._isamples = np.empty([0])
        self._isamples_in_confidence = np.empty([0])
        self._isamples_out_confidence = np.empty([0])
        self._isamples_synthetic = np.empty([0])
        
        
        # log-likelihood
        self.lnprobability = lnprobability
        self._ilnprobability = np.empty([0])

        
        # chi2 and n_orbit
        self._iorbit = None
        self._n_iorbit = 0.        
        self._ichi2_in_confidence = np.empty([0])
        self._n_iorbit_in_confidence = 0.                
        self._ichi2_out_confidence = np.empty([0])
        self._n_iorbit_out_confidence = 0.
        

        # Load observations
        self._ob, self._er = get_planet_data(data)
        if self._ob is None and self._er is None:
            try:
                #FileHandler.__init__(self,self._dataFilePath,addContent=[])
                st_temp = fh(data)
                (self._ob, self._er) = st_temp.getInfo()        
            except:
                raise Exception("Crash during the load of your data. Is the path correct ? --> {}".format(data))
          
        if len(self._ob.values()[0]) == 0:
            raise Exception("The file '{}' seems to be empty.".format(data))
        
        self._nob = len(self._ob['timePosition'])        
        self._ob['timePositionJD'] = [Time(self._ob['timePosition'][k],format='iso',scale='utc').jd for k in range(self._nob)]
       
        # Other arguments
        self._outputpath = outputPath          
        self._pKey = ['semiMajorAxis','eccentricity','inclinaison', 'longitudeAscendingNode', 'periastron', 'periastronTime']            
        self._nwalker, self._walker_length, self._nparameters = self.markovchain.shape             
        self._dof = self._nob*2-self._nparameters
        self._length = -1
        self._dim = self.markovchain.shape[2]        
        self._best = None        
        self.valMax = None
        self.confidenceInterval = None                 
        
    # --------
    # Methods
    # --------          
    @property
    def isamples(self):          
        return self._isamples

    @property
    def ilnprobability(self):          
        return self._ilnprobability           

    @property
    def ob(self):          
        return self._ob 
        
    @property
    def er(self):          
        return self._er         
    # ------------------------------------------------------------------------------------------------------------------------------------------        
    def independent_samples(self, burnin=0.5, verbose=True):
        """
        Extract independant samples from the Markov chain.
        
        """
        #if burnin is None:
        #    burnin = self._burnin
        start = int(np.floor(burnin*self._walker_length))
        self._isamples = self.markovchain[:,start:,:].reshape((-1,6))
        self._isamples_synthetic = self.markovchain_synthetic[:,start:,:].reshape((-1,6))
        
        if self.lnprobability is not None:
            self._ilnprobability = self.lnprobability[:,start:].reshape((-1))
            temp = np.unique(self._ilnprobability, return_index=True)
            self._iorbit = (temp[0][::-1],temp[1][::-1])
            self._n_iorbit = temp[0].shape[0]
            
        self._length = self._isamples.shape[0]
        
        if verbose:
            print 'Independent samples have been created and stored in the attribut: isamples.'
         

    # ------------------------------------------------------------------------------------------------------------------------------------------        
    def reset(self):
        """
        Clear the isamples, lnprobability, best, valMax, confidenceInterval and
        related attributs.
        
        """
        self._isamples = np.empty([0])
        self._isamples_synthetic = np.empty([0])
        self._ilnprobability = np.empty([0])
        self._iorbit = None
        self._length = -1
        self._isamples_in_confidence = np.empty([0])
        self._ichi2_in_confidence = np.empty([0])
        self._n_iorbit_in_confidence = 0.
        self._n_iorbit = 0.
        self._best = None
        self._isamples_out_confidence = np.empty([0])
        self._ichi2_out_confidence = np.empty([0])
        self._n_iorbit_out_confidence = 0.
        self.valMax = None
        self.confidenceInterval = None        
        
        
        
    # ------------------------------------------------------------------------------------------------------------------------------------------        
    def showPDFCorner(self, bins=50, labels=None, back_to_synthetic_elements=False, 
                      save=False, **kwargs):
        """ 
        Return a corner plot (base on triangle.py package) for the Markov chain.

        Parameters
        ----------
        bins: int (optional)
            The number of bins used to construct the posterior distribution.

        labels: list of str (optional)
            The x-axis labels.
            
        back_to_synthetic_elements : boolean (optional)
            If True and only if synthetic was True, the corner plot is returned 
            in terms of synthetic elements.
        
        save: bookea (optional)
            If True, the corner plot is saved in the output repository.
            
        kwargs: dict (optional)
            Additional parameters are passed to triangle.corner().
        
        """
        if back_to_synthetic_elements:
            if self._synthetic:            
                isamples = self._isamples_synthetic
            else:
                print 'The parameter -synthetic- was set to {} during the MCMC run.'.format(self._synthetic)
                print 'Therefore, isamples is used instead of isamples_synthetic.'
                isamples = self._isamples
                back_to_synthetic_elements = False
        else:
            isamples = self._isamples
        #else:
        #    if len(isamples.shape) > 2 and isamples.shape[2] == 6: 
        #        isamples = isamples.reshape((-1,6))
            
        if isamples.shape[0] == 0:
            print 'It seems that the isamples attribut is empty. You should first run independent_samples().'
            return None       


        if labels is None:
            if back_to_synthetic_elements and self._synthetic and self._priors['which']=='alternative':
                labels = [r"$\log(P)$",r"$e$",r"$\cos(i)$",r"$\Omega$ (deg)",r"$\omega$ (deg)","$t_p$ (JD)"]                
            elif back_to_synthetic_elements and self._synthetic and self._priors['which']=='Pueyo':
                labels = [r"$\log(P)$",r"$e$",r"$\cos(i)$",r"$\omega+\Omega$ (deg)",r"$\omega-\Omega$ (deg)","$t_p$ (JD)"]
            elif back_to_synthetic_elements and self._synthetic:
                labels = [r'$p_{}$'.format(j) for j in range(self._dim)] 
            else:
                labels = ["$a$ (AU)","$e$","$i$ (deg)","$\Omega$ (deg)","$\omega$ (deg)","$t_p$ (JD)"]
         
        fig = triangle.corner(isamples, labels=labels, bins=bins, **kwargs)    
        

        if save:
            plt.savefig(self._outputpath+'corner_plot.pdf')
            plt.close(fig)
        else:
            plt.show()   


    # ------------------------------------------------------------------------------------------------------------------------------------------                
    def showWalk(self, burnin=0., back_to_synthetic_elements=True,
                 figsize=(10,10), labels=None, save=False, **kwargs):
        """
        Illustration of the walk of each walker for all model parameters.
        
        Parameters
        ----------
        burnin: float (optional, default=0.0)
            The fraction of a walker which is discarded. (default: "None")

        back_to_synthetic_elements : boolean (optional)
            If True and only if synthetic was True, the walk plot is returned 
            in terms of synthetic elements.
            
        figsize: tuple of 2 floats
            The size of the output figure. (default: "(10,10)")
            
        labels: list of str (optional)
            The y-axis labels.            
        
        save: boolean (optional)
            If True, the walk plot is saved in the output repository.
            
        kwargs: dict
            Additional parameters are passed to matplotlib.pylab.plot()
            
        """                
        
        # Chain
        if back_to_synthetic_elements:
            chain_to_work_on = self.markovchain_synthetic.copy()
        else:
            chain_to_work_on = self.markovchain.copy()
        
        
        if burnin == 0.:
            chain = chain_to_work_on
        elif type(burnin) is tuple:
            bounds = [int(np.floor(b*self._walker_length)) for b in burnin]
            chain = chain_to_work_on[:,bounds[0]:bounds[1], :]
        else:
            chain = chain_to_work_on[:,int(np.floor(burnin*self._walker_length)):, :]
          
        # Figure          
        if labels is None:
            if back_to_synthetic_elements and self._synthetic and self._priors['which']=='alternative':
                labels = [r"$\log(P)$",r"$e$",r"$\cos(i)$",r"$\Omega$ (deg)",r"$\omega$ (deg)","$t_p$ (JD)"]                
            elif back_to_synthetic_elements and self._synthetic and self._priors['which']=='Pueyo':
                labels = [r"$\log(P)$",r"$e$",r"$\cos(i)$",r"$\omega+\Omega$ (deg)",r"$\omega-\Omega$ (deg)","$t_p$ (JD)"]
            elif back_to_synthetic_elements and self._synthetic:
                labels = [r'$p_{}$'.format(j) for j in range(self._dim)] 
            else:
                labels = ["$a$ (AU)","$e$","$i$ (deg)","$\Omega$ (deg)","$\omega$ (deg)","$t_p$ (JD)"] 
        
        #plt.clf()
        fig, axes = plt.subplots(6, 1, sharex=True, figsize=figsize)
        
        axes[self._dim-1].set_xlabel("step number",fontsize=kwargs.get('fontsize',10))
        for j in range(self._dim):            
            axes[j].yaxis.set_major_locator(MaxNLocator(5))
            axes[j].set_ylabel(labels[j], fontsize=kwargs.pop('fontsize',10))
            axes[j].plot(chain[:,:,j].T, color=kwargs.pop('color','k'), alpha=kwargs.pop('alpha',0.4),**kwargs)
        
        
        fig.tight_layout(h_pad=0.0)
        if save:
            plt.savefig(self._outputpath+'walk_plot.pdf')
            plt.close(fig)
        else:
            plt.show()            

    # ------------------------------------------------------------------------------------------------------------------------------------------                
    def confidence(self, cfd=68.27, bins = 100, gaussianFit=False, verbose=False,
                   save = False, **kwargs):
        """
        Determine the highly probable value for each model parameter, as well 
        as the 1-sigma confidence interval.
        
        Parameters
        ----------
        cfd: float (optional)
        
        bins: int (optional)
            The number bins used to sample the posterior distribution.
            
        gaussianFit: boolean (optional)
        
        verbose: boolean (optional)
            If True, additional information are displayed.
            
        save: boolean (optional)
            If True, a txt file with the results is saved in the output repository.
            
        kwargs: dict
        
        """
        if self._isamples.shape[0] == 0:
            print 'No independant samples found. You should first run the method: independent_samples().'
            return None, None        
        
        title = kwargs.pop('title',None)                
        #output_file = kwargs.pop('filename','confidence.txt')
        import matplotlib.pyplot as plt
        
        try:
            l = self._isamples.shape[1]        
        except Exception:
            l = 1
         
        #confidenceInterval = dict()
        confidenceInterval = np.zeros([6,2])
        #val_max = dict()
        val_max = np.zeros(6)
        #pKey = ['p0','p1','p2','p3','p4','p5']#['r','theta','f']
        
        if cfd == 100:
            cfd = 99.9
            
        #########################################    
        ##  Determine the confidence interval  ##
        #########################################
        if gaussianFit:
            mu = np.zeros(3)
            sigma = np.zeros_like(mu)
            
        for j in range(l):              
            
             
            label_file = ['p0','p1','p2','p3','p4','p5']    
            label = label_file#[r'$\Delta r$',r'$\Delta \theta$',r'$\Delta f$']
            
            plt.figure()
            n, bin_vertices, patches = plt.hist(self._isamples[:,j],bins=bins,
                                                histtype=kwargs.get('histtype','step'), 
                                                edgecolor=kwargs.get('edgecolor','blue'))
            bins_width = np.mean(np.diff(bin_vertices))
            surface_total = np.sum(np.ones_like(n)*bins_width * n)
            n_arg_sort = np.argsort(n)[::-1]
            
            test = 0
            pourcentage = 0
            for k,jj in enumerate(n_arg_sort):
                test = test + bins_width*n[jj]
                pourcentage = test/surface_total*100.
                if pourcentage > cfd:
                    if verbose:
                        print 'pourcentage for {}: {}%'.format(label_file[j],pourcentage)
                    break
            n_arg_min = n_arg_sort[:k].min()
            n_arg_max = n_arg_sort[:k+1].max()
            if n_arg_min == 0:
                n_arg_min += 1                
            if n_arg_max == bins:
                n_arg_max -= 1            
            
            #val_max[pKey[j]] = bin_vertices[n_arg_sort[0]]+bins_width/2.
            val_max[j] = bin_vertices[n_arg_sort[0]]+bins_width/2.
            #confidenceInterval[pKey[j]] = np.array([bin_vertices[n_arg_min-1],bin_vertices[n_arg_max+1]]-val_max[pKey[j]])
            confidenceInterval[j,:] = np.array([bin_vertices[n_arg_min-1],bin_vertices[n_arg_max+1]]-val_max[j])
                            
            arg = (self._isamples[:,j]>=bin_vertices[n_arg_min-1])*(self._isamples[:,j]<=bin_vertices[n_arg_max+1])            
            
            n2, bins2, patches2 = plt.hist(self._isamples[arg,j],bins=bin_vertices,
                                           facecolor=kwargs.get('facecolor','green'),
                                           edgecolor=kwargs.get('edgecolor','blue'), histtype='stepfilled')
            
            #plt.plot(np.ones(2)*val_max[pKey[j]],[0,n[n_arg_sort[0]]],'--m')
            plt.plot(np.ones(2)*val_max[j],[0,n[n_arg_sort[0]]],'--m')
            plt.xlabel(label[j]) 
            plt.ylabel('Counts')
            
            
            if gaussianFit:
                import matplotlib.mlab as mlab
                from scipy.stats import norm
                plt.figure()
                plt.hold('on')
        
                (mu[j], sigma[j]) = norm.fit(self._isamples[:,j])
                n_fit, bins_fit = np.histogram(self._isamples[:,j], bins, normed=1)
                a, b, c = plt.hist(self._isamples[:,j], bins, normed=1,
                                   facecolor=kwargs.get('facecolor','green'),
                                   edgecolor=kwargs.get('edgecolor','blue'), 
                                   histtype=kwargs.get('histtype','step'))
                y = mlab.normpdf( bins_fit, mu[j], sigma[j])
                l = plt.plot(bins_fit, y, 'r--', linewidth=2, alpha=0.7) #/y.max()*n[n_arg_sort[0]]
                
                plt.xlabel(label[j]) 
                plt.ylabel('Counts')
                if title is not None:
                    plt.title(title+'   '+r"$\mu$ = {:.4f}, $\sigma$ = {:.4f}".format(mu[j],sigma[j]), fontsize=10, fontweight='bold')
            else:
                if title is not None:            
                    plt.title(title, fontsize=10, fontweight='bold')
        
                        
            if save:
                if gaussianFit:
                    plt.savefig('confi_hist_{}_gaussFit.pdf'.format(label_file[j]))
                else:
                    plt.savefig('confi_hist_{}.pdf'.format(label_file[j]))
            plt.show()

        if gaussianFit:
            return (mu,sigma)
        else:
            self.valMax = val_max
            self.confidenceInterval = confidenceInterval
            return (val_max,confidenceInterval)    


    # ------------------------------------------------------------------------------------------------------------------------------------------ 
    def best_solution(self):
        """
        If the log-likelihood associated to the Markov chain was given, the best
        solution in terms of reduced chi2 is returned.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        out : (set of model parameters, corresponding reduced chi2)
        
        """
        if self.lnprobability is None:
            print '''The method -best_solution()- can only be called if the
log-likelihood class attribut (lnprobability) has been given.'''
            return None
        elif self.isamples.shape[0] == 0:
            print 'No independant samples found. You should first run the method: independent_samples().'
            return None
            
        index = self._ilnprobability.argmax()
        err = np.concatenate((self._er['raError'],self._er['decError']))
        inv_sigma2 = 1.0/(err**2)        
        chi2 = -2*self._ilnprobability[index] + np.sum(np.log(inv_sigma2))
        self._best =  self._isamples[index,:], chi2/self._dof    
        return self._best


    # ------------------------------------------------------------------------------------------------------------------------------------------        
    def solution_in_confidence(self, valMax=None, confidenceInterval=None, verbose=True):
        """
        If the log-likelihood associated to the Markov chain was given, the 
        isamples are separated in two categories: those for which the parameters
        are all in the confidence intervals and those for which they are out.
        
        Parameters
        ----------
        
        verbose: boolean (optional)
        
        """  
        if self._isamples.shape[0] == 0:
            print 'No independant samples found. You should first run the method: independent_samples().'
            return None        
        
        if self.lnprobability is None:
            print '''The method -solution_in_confidence()- can only be called 
if the log-likelihood was given when instanciate the BayesianInference object.'''
            return None                        
           
        if valMax is None and self.valMax is not None:   
            valMax = self.valMax
        elif valMax is None and self.valMax is None:
            print 'No confidence interval found. You should first run the method: confidence()'
            return None
            
        if confidenceInterval is None and self.confidenceInterval is not None:
            confidenceInterval = self.confidenceInterval           
           
        err = np.concatenate((self._er['raError'],self._er['decError']))
        inv_sigma2 = 1.0/(err**2)        
        chi2_all = (-2*self._iorbit[0] + np.sum(np.log(inv_sigma2)))/self._dof

        bound_inf = valMax+confidenceInterval[:,0]
        bound_sup = valMax+confidenceInterval[:,1]
        
        sol_sup_than_bound_inf = np.all(np.tile(bound_inf,(self._n_iorbit,1))<=self._isamples[self._iorbit[1],:],axis=1)
        sol_inf_than_bound_sup = np.all(np.tile(bound_sup,(self._n_iorbit,1))>=self._isamples[self._iorbit[1],:],axis=1)
        

        sol_in_confidence = np.logical_and(sol_sup_than_bound_inf,sol_inf_than_bound_sup)#sol_sup_than_bound_inf*sol_inf_than_bound_sup
        self._n_iorbit_in_confidence = sol_in_confidence[sol_in_confidence].shape[0]
        self._isamples_in_confidence = self._isamples[self._iorbit[1][sol_in_confidence]]
        self._ichi2_in_confidence = chi2_all[sol_in_confidence]

        sol_out_confidence = np.logical_not(sol_in_confidence)
        self._n_iorbit_out_confidence = sol_out_confidence[sol_out_confidence].shape[0]
        self._isamples_out_confidence = self._isamples[self._iorbit[1][sol_out_confidence]]
        self._ichi2_out_confidence = chi2_all[sol_out_confidence]            
        
        if verbose:
            print 'Solutions in (resp. out of) confidence intervals have been stored in attribut isamples_in_conficence (resp. isamples_out_confidence).'
        
        #return self._isamples_in_confidence, self._ichi2_in_confidence
        
        

        
    # ------------------------------------------------------------------------------------------------------------------------------------------                
    def highlyProbableSolution(self, priors, nSolutions=1, chi2_max=1, 
                               nOrbit=None, in_confidence=None, verbose=False):
        """ 
        Search for the highly probable solution(s) in the Markov chain.
        
        Two methods can be adopted:
        #1 One needs only 1 solution (:param nSolution: set to 1) with a 
        reduced chi2 smaller than a given threshold (:param condition:).
        
        #2 One needs n orbits (:param nOrbit:) with a reduced chi2 smaller 
        than a given threshold.
        
        Parameters
        ----------
        
        priors
        nOrbit
        nSolution
        chi2_max
        valMax
        confidenceInterval             
        verbose
        """  
        ### TODO: Renommer variables
        condition = chi2_max
        nSolution = nSolutions
        valMax = self.valMax
        confidenceInterval = self.confidenceInterval
        ### 
        
        # Dependencies
        if self.isamples.shape[0] == 0:
            print 'No independant samples found. You should first run the method: independent_samples().'
            return None, None
        
        if self.lnprobability is not None and self._isamples_in_confidence.shape[0]==0:
            print 'No independant samples in confidence found. You should first run the method: solution_in_confidence().'
            return None, None

 
        if in_confidence is not None:#if valMax is not None and confidenceInterval is not None:
            criteria_number = 2
        else:
            criteria_number = 1
  
                      
        if nOrbit is None:
            nOrbit = np.min([500,self._length])

        ################################# 
        ##  IF LNPROBABILITY IS GIVEN  ##
        #################################
        # TODO: valMax et confidenceInterval ne sont plus utiliser que dans le if: peut être supprimer l'appel à solution_in_confidence et imposer que self._valMax et self._confidenceInterval existent.
        
        if self.lnprobability is not None:                        
            if in_confidence:
                #if self._isamples_in_confidence.shape[0] == 0:
                #    BayesianInference.solution_in_confidence(valMax, confidenceInterval)
                
                p_condition_ok = self._isamples_in_confidence[self._ichi2_in_confidence<condition,:]
                chi2_condition_ok = self._ichi2_in_confidence[self._ichi2_in_confidence<condition]
            elif in_confidence is not None and not in_confidence:
                #if self._isamples_out_confidence.shape[0] == 0:
                #    BayesianInference.solution_in_confidence(valMax, confidenceInterval)  

                p_condition_ok = self._isamples_out_confidence[self._ichi2_out_confidence<condition,:]
                chi2_condition_ok = self._ichi2_out_confidence[self._ichi2_out_confidence<condition]       
            elif in_confidence is None:
                p_condition_ok = np.vstack((self._isamples_in_confidence[self._ichi2_in_confidence<condition,:],
                                            self._isamples_out_confidence[self._ichi2_out_confidence<condition,:]))                                                                                                                                   
                chi2_condition_ok = np.concatenate((self._ichi2_in_confidence[self._ichi2_in_confidence<condition],
                                               self._ichi2_out_confidence[self._ichi2_out_confidence<condition]))

            n_condition_ok = p_condition_ok.shape[0]            
            which = np.random.choice(range(n_condition_ok), np.amin([nSolution,n_condition_ok]), replace=False)                
            p_res = p_condition_ok[which]
            chi2_res = chi2_condition_ok[which]
            
            if verbose:
                print '{} solutions with a reduced chi2 <= {} have been found.'.format(p_res.shape[0], chi2_max)
            return p_res, chi2_res
        #else:
        #    print 'Confidence interval not found. You should start running the method confidence()'
        #    return None, None
            
                            
            
            
        #####################################   
        ##  IF LNPROBABILITY IS NOT GIVEN  ##
        #####################################                
        # ONLY 1 SOLUTION 
        ### TODO: la partie du code où seule une solution est demandée devrait être intégrée à la partie où N sont demandées    
        if nSolution == 1:
            tt = self._isamples[np.random.choice(range(self._length), nOrbit, replace=False),:]

            chi2Orbit = np.inf
            temp = np.inf
            chi2_min = np.inf
            
            for j in range(nOrbit):
                criteria = [False for i in range(criteria_number)]
                
                orbit = o.Orbit(semiMajorAxis = tt[j,0],\
                            eccentricity = tt[j,1],\
                            inclinaison = tt[j,2],\
                            longitudeAscendingNode = tt[j,3],\
                            periastron = tt[j,4],\
                            periastronTime = tt[j,5],\
                            dStar = priors['starDistance'], starMass = priors['starMass'])
                                                      
                temp = orbit.chiSquare(self._ob,self._er)/self._dof
                
                criteria[0] = temp < condition
                if criteria[0] and criteria_number == 1:
                    chi2Orbit = temp
                    parameterMin = tt[j,:]
                    if verbose:
                        print('New chi2r: {}'.format(chi2Orbit)) 
                    break
                elif criteria[0] and criteria_number ==2:                    
                    p = tt[j,:]                   
                    if in_confidence and all((valMax+confidenceInterval[:,1])>=p) and all((valMax+confidenceInterval[:,0])<=p):
                        chi2Orbit = temp
                        parameterMin = p
                        if verbose:
                            print('New chi2r: {}'.format(chi2Orbit)) 
                        break
                    elif any((valMax+confidenceInterval[:,1])<p) or any((valMax+confidenceInterval[:,0])>p):
                        chi2Orbit = temp
                        parameterMin = p
                        if verbose:
                            print('New chi2r: {}'.format(chi2Orbit)) 
                        break                    
                elif temp <= chi2_min:
                    chi2_min = temp                    

            try:
                return parameterMin, chi2Orbit
            except:
                print 'No solution with chi2r <= {} have been found in the sample of {} tested orbits.\nThe minimum chi2r found in this sample is {}'.format(condition,nOrbit,chi2_min)              
                return None, None
        
        # A SET OF SOLUTIONS 
        elif nSolution > 1: 
            chi2_min = np.inf
            nOrbit = np.amin([int(5e02*nSolution),self._length])         
            
            tt = self._isamples[np.random.choice(range(self._length), nOrbit, replace=False),:]            
            
            chi2Orbit = np.ones(nSolution)*np.inf
            temp = np.inf

            k = 0  
            for j in range(nOrbit):
                criteria = [False for i in range(criteria_number)]
                
                orbit = o.Orbit(semiMajorAxis = tt[j,0],\
                            eccentricity = tt[j,1],\
                            inclinaison = tt[j,2],\
                            longitudeAscendingNode = tt[j,3],\
                            periastron = tt[j,4],\
                            periastronTime = tt[j,5],\
                            dStar = priors['starDistance'], starMass = priors['starMass'])
     
                temp = orbit.chiSquare(self._ob,self._er)/self._dof                
                
                criteria[0] = temp < condition
                if criteria[0] and criteria_number == 1:
                    chi2Orbit[k] = temp
                    if k == 0:
                        parameterMin = tt[j,:]
                    else:
                        parameterMin = np.vstack((parameterMin,tt[j,:]))
                        
                    if verbose:    
                        print('Stored chi2r: {}'.format(chi2Orbit[k]))
                    k += 1
                elif criteria[0] and criteria_number ==2:
                    p = tt[j,:]
                    
                    if in_confidence:
                        cond = [all((valMax+confidenceInterval[:,1])>=p),
                                all((valMax+confidenceInterval[:,0])<=p)]
                    else:
                        cond = [any((valMax+confidenceInterval[:,1])<p),
                                any((valMax+confidenceInterval[:,0])>p)]
                                
                    if cond[0] and cond[1]:
                        chi2Orbit[k] = temp
                        if k == 0:
                            parameterMin = p
                        else:
                            parameterMin = np.vstack((p,parameterMin))
                        if verbose:    
                            print('Stored chi2r: {}'.format(chi2Orbit[k]))
                        k += 1                        
                elif temp <= chi2_min:
                    chi2_min = temp
                    

                if k >= nSolution:
                    return (parameterMin, chi2Orbit)
            
            try:
                return (parameterMin, chi2Orbit)
            except:
                print 'No solutions with chi2r <= {} have been found in the sample of {} tested orbits.\nThe minimum chi2r found in this sample is {}'.format(condition,nOrbit,chi2_min)
                return (None, None)

#    # ------------------------------------------------------------------------------------------------------------------------------------------                
#    def showOrbit(self, p, priors, link = True, figsize = (15,15), lim = [], title = None, display = True, save = (False,''), output = False):
#        """
#        Return
#        
#        Parameters
#        ----------
#        p
#        priors
#        link
#        figsize
#        lim
#        title
#        display
#        save
#        output
#        """
#        #p = [67.95,0,22.06,55.5,0,2451259.5] # HR8799b
#        orbit = o.Orbit(semiMajorAxis=p[0],\
#                    eccentricity=p[1],\
#                    inclinaison=p[2],\
#                    longitudeAscendingNode=p[3],\
#                    periastron=p[4],\
#                    periastronTime=p[5],\
#                    starMass=priors['starMass'],\
#                    dStar = priors['starDistance'])
#        
#        #lim = [[-0.30,-0.10],[-0.42,-0.22]] # betapicb
#        #lim = [[1.3,1.7],[0.65,1.05]] # HR8799b
#        orbitLine = orbit.showOrbit(unit='arcsec',\
#                                addPoints=[self._ob['raPosition'],self._ob['decPosition'],self._er['raError'],self._er['decError']],\
#                                lim=lim,\
#                                addPosition=self._ob['timePosition'],\
#                                link = link,\
#                                figsize = figsize,\
#                                title = title,\
#                                display = display,\
#                                save=save,\
#                                output = output)    
#
#        return orbitLine

    # ------------------------------------------------------------------------------------------------------------------------------------------                
    def showSolutions(self, priors, p, chi2r, pchi2r0=None, addPosition=None,
                      best_options=None, title_options=None, save_options=None,
                      positions=False, **kwargs):
        """
        """    
        pointOfView='earth' # TODO: BUG lorsque fixé à "face-on"
                
        if pchi2r0 is not None:         
            p0, chi2r0 = pchi2r0            
            p = np.vstack((p,p0))
            chi2r = np.hstack((chi2r,chi2r0))
           
        if len(p.shape) == 1:
            p = np.vstack((p,p))
            chi2r = np.array([chi2r,chi2r])
        
        indexmin = np.argmin(chi2r)        
        chi2rmin = np.amin(chi2r)
        chi2rmax = np.amax(chi2r)
        chi2diff = chi2rmax-chi2rmin if chi2rmax-chi2rmin != 0 else 1
        
        if addPosition is not None:
            for new in addPosition:
                self._ob['timePosition'].append(new)
        
        xmin = 0
        ymax = 0        
        
        # EXTRACT NON MAIN PLOT PARAMETERS FROM KWARGS        
        #best_kwargs = {}
        #for key in kwargs.keys():
        #    if key.find('best') != -1:
        #        best_kwargs[key[5:]] = kwargs[key]
        
        
        # PLOT PARAMETERS FROM **KWARGS WHICH NEED TO BE IN THE LOOP       
        figsize=kwargs.pop('figsize',(10,10))
        color = kwargs.pop('color',(0,97/255.,1))        
        linewidth = kwargs.pop('linewidth',1)
        
        # Color
        if color == 'r':
            color = (1,0,0)
        elif color == 'g':
            color = (0,1,0)
        elif color == 'b':
            color = (0,0,1)
        
        # FIGURE
        plt.figure(figsize=figsize)
        plt.hold(True)

        lim = kwargs.pop('lim',None)
        if lim is not None:
            plt.xlim((lim[0][0],lim[0][1]))
            plt.ylim((lim[1][0],lim[1][1]))   
        
        # Ticks
        xticks = kwargs.pop('xticks',None)
        yticks = kwargs.pop('yticks',None)
        if xticks is not None:
            plt.xticks(xticks,xticks,size=kwargs.pop('xticks_size',20))
        else:
            plt.xticks(size=kwargs.pop('xticks_size',20))
        
        if yticks is not None:
            plt.yticks(yticks,yticks,size=kwargs.pop('yticks_size',20))
        else:
            plt.yticks(size=kwargs.pop('yticks_size',20))            
        
        #
        plt.axes().set_aspect('equal')
        plt.gca().invert_xaxis() # Then, East is left
        
        if pointOfView == 'earth':
            unit = 'arcsec'            
        elif pointOfView == 'face-on':
            unit = 'au'           

        for j in range(p.shape[0]):
            orbit = o.Orbit(semiMajorAxis=p[j][0],\
                    eccentricity=p[j][1],\
                    inclinaison=p[j][2],\
                    longitudeAscendingNode=p[j][3],\
                    periastron=p[j][4],\
                    periastronTime=p[j][5],\
                    starMass=priors['starMass'],\
                    dStar = priors['starDistance'])
            
            if pointOfView == 'face-on':
                angle = p[j][3]/180.*np.pi
            else:
                angle = 0
            
            rotateMatrix = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
                                        
            if j != indexmin:
                addPoints = [self._ob['raPosition'],self._ob['decPosition'],
                             self._er['raError'],self._er['decError']]

#                orbitLine, point = orbit.showOrbit(pointOfView=pointOfView,\
#                                                    unit=unit,\
#                                                    addPoints=addPoints,\
#                                                    lim=lim,\
#                                                    addPosition=self._ob['timePosition'],\
#                                                    figsize = figsize,\
#                                                    title = None,\
#                                                    output = 'onlyAll')
                orbitLine, point = orbit.showOrbit(unit='arcsec',
                                        addPoints=addPoints,
                                        addPosition=self._ob['timePosition'],
                                        output = 'onlyAll')                                                    
                
                factor = ((chi2rmax-chi2r[j])/chi2diff) 
                #print factor
                coeff_alpha = 0.2 + factor*0.5

                color_comp = (1-color[0],1-color[1],1-color[2])                    
                #color_new = (color[0]*factor,color[1]*factor,color[2]*factor)
                color_new = (color[0]*factor+color_comp[0]*(1-factor),
                             color[1]*factor+color_comp[1]*(1-factor),
                             color[2]*factor+color_comp[2]*(1-factor))
                
                                
                orbitRotate = np.transpose(np.dot(rotateMatrix,np.array([orbitLine['raPosition'],orbitLine['decPosition']])))                                                                                  
                plt.plot(orbitRotate[:,0], orbitRotate[:,1], color=color_new, linewidth=linewidth*factor, alpha=coeff_alpha)
               
                if positions:
                    pointRotate = np.transpose(np.dot(rotateMatrix,np.transpose(point)))
                    plt.plot(pointRotate[:self._nob,0],pointRotate[:self._nob,1],'o', color=(0,97/255.,1), markersize = 2, markeredgecolor = 'none')
                    plt.plot(pointRotate[self._nob:,0],pointRotate[self._nob:,1],'o', color=(0/255.,128/255.,0/255.), markersize = 2, markeredgecolor = 'none')
                
                if np.amin(orbitRotate[:,0]) < xmin:
                    xmin = np.amin(orbitRotate[:,0])
                if np.amax(orbitRotate[:,1]) > ymax:
                    ymax = np.amax(orbitRotate[:,1])                
            else:
                addPoints = [self._ob['raPosition'],self._ob['decPosition'],
                             self._er['raError'],self._er['decError']]

                orbitLine, point = orbit.showOrbit(pointOfView=pointOfView,\
                            unit=unit,\
                            addPoints=addPoints,\
                            lim=lim,\
                            addPosition=self._ob['timePosition'],\
                            figsize = figsize,\
                            title = None,\
                            output = 'onlyAll')
                pointRotate0 = np.transpose(np.dot(rotateMatrix,np.transpose(point)))
                orbitRotate0 = np.transpose(np.dot(rotateMatrix,np.array([orbitLine['raPosition'],orbitLine['decPosition']])))                
                #plt.plot(orbitLine['raPosition'],orbitLine['decPosition'],color=(0,97/255.,1), linewidth=2, alpha = 1)

                                                                                                     
        # Plot of the min chi2r solution
        if best_options is not None:
            plt.plot(orbitRotate0[:,0], orbitRotate0[:,1], **best_options)#, color=best_color, linewidth=1.5, alpha = 1)
              
        # Plot data and error bars                          
        if pointOfView == 'earth':
            plt.xlabel(r"$\Delta$ RA (arcsec)", fontsize=20)
            plt.ylabel(r"$\Delta$ DEC (arcsec)", fontsize=20)
            plt.plot(self._ob['raPosition'], self._ob['decPosition'],'ko')
            plt.errorbar(self._ob['raPosition'],self._ob['decPosition'], xerr = self._er['raError'], yerr = self._er['decError'], fmt = 'k.')
        else:
            plt.xlabel("x (AU)", fontsize=20)
            plt.ylabel("y (AU)", fontsize=20)
            plt.arrow(0,0,0.55*xmin,0, width = 0.05, head_width=0.5, head_length=1, fc='k', ec='k',alpha = 0.3, label = 'test')
            plt.plot(0,0,'ok',markersize=5)
            plt.text(0.52*xmin,0.05*ymax, 'To Earth', fontsize = 14)              
            plt.plot(pointRotate0[:self._nob,0],pointRotate0[:self._nob,1],'o',color=(1,20/255.,47/255.), markersize = 7,markeredgecolor = 'none')
            plt.plot(pointRotate0[self._nob:,0],pointRotate0[self._nob:,1],'o',color=(0/255.,128/255.,0/255.), markersize = 7,markeredgecolor = 'none')
            

            
        # Define the title
        if title_options is not None:        
            plt.title(title_options.pop('title',''), **title_options)
        
        # Save
        if save_options is not None:             
            filename = save_options.pop('filename','figure')+'.'+save_options.get('format','pdf')
            plt.savefig(filename, **save_options)
        
        return None