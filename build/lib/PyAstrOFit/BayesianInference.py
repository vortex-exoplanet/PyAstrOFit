# -*- coding: utf-8 -*-


import numpy as np
import pickle

from astropy.time import Time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from StatisticsMCMC import StatisticsMCMC as st
from FileHandler import FileHandler as fh
import Orbit as o


__all__ = ["BayesianInference"]


packageName = 'PyAstrOFit'


class BayesianInference(object):
    """ Tools to perform Bayesian Inference and Markov chains representation.
    
    :param isamples:
        The Markov chain obtained from a emcee run. This attribut has the shape
        (nwalker, steps, dim), where nwalker represents the number of walker, 
        steps the number of steps per walker and dim the number of model
        parameters.
    
    :param data:
        The path towards the data file.
        
    :param outputPath:
        The path towards the stored result files.
                                        
    """
    
    # -----------------------
    # Class attributs
    # -----------------------        
    
    
    def __init__(self,  isamples,\
                        data,\
                        priors,\
                        outputPath = '',\
                        burnin = 0.0,\
                        synthetic = False,\
                        save = False):
                            
        """ The constructor of the class """
        # -----------------------
        # Preamble instructions
        # -----------------------        
        # NONE        
        
        # -----------------------------
        # Main part of the constructor
        # -----------------------------
        if data == 'betapicb':
            self._dataFilePath = 'MoucMouc/res/exo/betaPicb.txt'
        elif data == 'hr8799b':
            self._dataFilePath = 'MoucMouc/res/exo/hr8799b.txt' 
        elif data == 'hr8799c':
            self._dataFilePath = 'MoucMouc/res/exo/hr8799c.txt'
        elif data == 'hr8799c_Pueyo':
            self._dataFilePath = 'MoucMouc/res/exo/hr8799c_Pueyo.txt'
        elif data == 'hr8799c_Pueyo_largeEr':
            self._dataFilePath = 'MoucMouc/res/exo/hr8799c_Pueyo_largeEr.txt'             
        elif data == 'hr8799c_withoutLBT':
            self._dataFilePath = 'MoucMouc/res/exo/hr8799c_withoutLBT.txt'  
        elif data == 'hr8799c_selection':
            self._dataFilePath = 'MoucMouc/res/exo/hr8799c_selection.txt'             
        elif data == 'hr8799d':
            self._dataFilePath = 'MoucMouc/res/exo/hr8799d.txt'             
        elif data == 'hr8799e':
            self._dataFilePath = 'MoucMouc/res/exo/hr8799e.txt'   
        elif data == 'orbit_test_0':
            self._dataFilePath = 'MoucMouc/res/exo/orbit_test_0.txt' 
        elif data == 'orbit_test_1':
            self._dataFilePath = 'MoucMouc/res/exo/orbit_test_1.txt' 
        elif data == 'orbit_test_2':
            self._dataFilePath = 'MoucMouc/res/exo/orbit_test_2.txt'
        elif data == 'orbit_test_3':
            self._dataFilePath = 'MoucMouc/res/exo/orbit_test_3.txt'             
        elif data == 'orbit_test_3_true':
            self._dataFilePath = 'MoucMouc/res/exo/orbit_test_3_true.txt' 
        elif data == 'orbit_test_3_largeEr':
            self._dataFilePath = 'MoucMouc/res/exo/orbit_test_3_largeEr.txt'             
        else:
            self._dataFilePath = data

        try:
            #FileHandler.__init__(self,self._dataFilePath,addContent=[])
            st = fh(self._dataFilePath)
            (self._ob, self._er) = st.getInfo()        
            if len(self._ob.values()[0]) == 0:
                raise Exception("The file '{}' seems to be empty.".format(self._dataFilePath))
            self._ob['timePositionJD'] = [Time(self._ob['timePosition'][k],format='iso',scale='utc').jd for k in range(len(self._ob['timePosition']))]
            self._nob = len(self._ob['timePositionJD'])
            self._dof = self._nob*2-6
        except:
            raise Exception("Crash during the load of your data. Is the path correct ? --> {}".format(self._dataFilePath))
          

        self._burnin = burnin
        if isamples == None:
            self._isamples = np.empty(0)
        elif isinstance(isamples,np.ndarray):
            if len(isamples.shape) == 3:
                length = isamples.shape[1]
                self._isamples = isamples[:,int(np.floor(self._burnin*length)):,:].reshape((-1,6))
            else:
                self._isamples = isamples
            self._runInfo = None
        else:
            # then the attribut -isamples- is a pickle object which contain the np.ndarray
            try:
                with open(isamples,'rb') as fileRead:
                    myPickler = pickle.Unpickler(fileRead)
                    res = myPickler.load()
                    #print(res)
                if len(res['isamples'].shape) == 3:
                    length = res['isamples'].shape[1]
                    self._isamples = res['isamples'][:,int(np.floor(self._burnin*length)):,:].reshape((-1,6))
                    self._isamples_unflat = res['isamples'][:,int(np.floor(self._burnin*length)):,:]
                else:
                    length = res['isamples'].shape[0]
                    self._isamples = res['isamples'][int(np.floor(self._burnin*length)):,:]
                self._runInfo = res['run_info']
            except:
                raise Exception("The file -{}- doesn't exist or isn't a pickle object.".format(isamples))
        
        self._priors = priors
        self._length = self._isamples.shape[0]      
        self._outputpath = outputPath          
        self._pKey = ['semiMajorAxis','eccentricity','inclinaison', 'longitudeAscendingNode', 'periastron', 'periastronTime']
        self._chainKepler = 0
        
    # --------
    # Methods
    # --------  
    @property
    def priors(self):          
        return self._runInfo['priors']

    # ------------------------------------------------------------------------------------------------------------------------------------------        
    def saveisamples(self, name):
        """
        """
        with open(self._outputpath+name,'wb') as fileSave:
            myPickler = pickle.Pickler(fileSave)
            myPickler.dump(self._isamples_unflat) 
        
        
    # ------------------------------------------------------------------------------------------------------------------------------------------        
    def runReport(self):
        """Display a short report about the run."""
        
        timeForConv = self._runInfo['duration konvergence'].seconds/3600.
        stepsForConvPerWalker = self._runInfo['konvergence']
        stepsForConv = self._runInfo['konvergence']*self._runInfo['nwalkers']
        initialState = self._runInfo['initialState']
        threads = self._runInfo['threads']
        
        print("")
        print("Time for convergence (in hour): {}".format(timeForConv))
        print("Steps for convergence (per walkers): {} ({})".format(stepsForConv,stepsForConvPerWalker))
        print("Initial state: {}".format(initialState))
        print("Number of threads: {}".format(threads))
        
    # ------------------------------------------------------------------------------------------------------------------------------------------        
    def showPDFCorner(self, isamples = None, burnin = None, truths = None, bins = None, labels = None, extents = None, synthetic = False, mass = 0, save = False):
        """ Return a corner plot (base on triangle.py package) for the Markov chain.
        
        :param burnin:
            The fraction of a walker which is discarded. (default: "0.0")
            
        :param truths:
            If "initial", the position of the initial state is depicted in
            the corner plot.
            
        :param bins:
            The number of bins used to construct the posterior distribution.
            
        :param extents:
            The x-axis bounds, :typ: list of tuples.
        
        :param save:
            If "True", the corner plot is saved in the output repository.
        
        """
        try:
            import triangle
        except ImportError:
            raise Exception("You should install the Triangle package: https://github.com/dfm/triangle.py")


        if isamples == None:
            chain = self._isamples.reshape((-1,6))
        else:
            if len(isamples.shape) > 2 and isamples.shape[2] == 6: 
                chain = isamples.reshape((-1,6))
            else:
                chain = isamples
        if truths == 'initial':
            truths = self._runInfo['initialState']
            
        if bins == None:
            bins = 50
            
        if synthetic:
            # Pueyo
            #print chain.shape
            temp = np.zeros_like(chain)
            from Sampler import toKepler#, toSynthetic, period, semimajoraxis
            for j in range(chain.shape[0]):
                #if j%chain.shape[0]/100 == 0:
                #    print j
                temp[j,:] = toKepler(chain[j,:],which = self._priors['which'],mass=mass, referenceTime=self._priors['referenceTime'])
            chain = temp
            self._isamples = chain
                
            
            #labels = [r"$\log(P)$",r"$e$",r"$\cos(i)$",r"$\omega+\Omega$ (deg)",r"$\omega-\Omega$ (deg)","$t_\omega$ (JD)"]
        #else:
        if labels is None:    
            labels = ["$a$ (AU)","$e$","$i$ (deg)","$\Omega$ (deg)","$\omega$ (deg)","$t_\omega$ (JD)"]

        if chain.shape[0] == 0:
            print("It seems that the chain is empty. Have you already run the MCMC ?")
        else:
            
            fig = triangle.corner(chain, labels=labels, truths=truths, extents= extents, bins = bins)    
        
        if save:
            plt.savefig(self._outputpath+'corner_plot.pdf')
            plt.close(fig)
        else:
            plt.show()   


    # ------------------------------------------------------------------------------------------------------------------------------------------                
    def showWalk(self, burnin = None, isamples = None, figsize = (10,10), labels = None, save = False):
        """Illustration of the walk of each walker for all model parameters.
        
        :param burnin:
            The fraction of a walker which is discarded. (default: "None")

        :param figsize:
            The size of the output figure. (default: "(10,10)")
            
        :param save:
            If "True", the walk plot is saved in the output repository.
        """
        
        if burnin == None:
            burnin = 0
            #burnin = self._burnin

        if isamples == None: 
                chain = self._isamples_unflat
                print chain.shape
        else:
            chain = isamples
                
        if labels is None:
            labels = ["$a$ (AU)","$e$","$i$ (deg)","$\Omega$ (deg)","$\omega$ (deg)","$t_\omega$ (JD)"]
        #plt.clf()
        fig, axes = plt.subplots(6, 1, sharex=True, figsize=figsize)
        for j in range(6):
            #axes[j].plot(self._chain[:,int(np.floor(burnin*self._iterations)):, j].T, color="k", alpha=0.4)
            axes[j].plot(chain[:,:,j].T, color="k", alpha=0.4)
            axes[j].yaxis.set_major_locator(MaxNLocator(5))
            axes[j].set_ylabel(labels[j])
        axes[j].set_xlabel("step number")
        fig.tight_layout(h_pad=0.0)
        if save:
            plt.savefig(self._outputpath+'walk_plot.pdf')
            plt.close(fig)
        else:
            plt.show()            

    # ------------------------------------------------------------------------------------------------------------------------------------------                
    def confidence(self, nBin = 1000, verbose = False, save = False):
        """Determine the highly probable value for each model parameter, as well as the 1-sigma confidence interval.
        
        :param nBin:
            The number bins used to sample the posterior distribution.
            
        :param save:
            If "True", a txt file with the results is saved in the output repository.
        """
        cfd = 68.  
        confidenceInterval = dict()  
        val_max = dict()
        #pourcentBothPart = dict()
        for j in range(6):            
            xHistF, binsF = np.histogram(self._isamples[:,j],bins = nBin,density=False)
            nHistF = float(np.sum(xHistF))
            arg_max = xHistF.argmax()
            height_max = xHistF[arg_max]
            val_max[self._pKey[j]] = binsF[arg_max]
            
            #pourcentBothPart[self._pKey[j]] = np.array([np.sum(xHistF[:arg_max])/nHistF,np.sum(xHistF[arg_max:])/nHistF])
            
            # method #1
            pourcentage = 0.
            k = 0
            while pourcentage < cfd:
                arg_bound = np.where(xHistF>height_max-k)[0]
                pourcentage = np.sum(xHistF[arg_bound])/nHistF*100.
                k += 1
            #print((height_max-(k-1))/float(height_max))                
            confidenceInterval[self._pKey[j]] = binsF[arg_bound[[0,-1]]]-val_max[self._pKey[j]]

        if save:
            ### Write inference results in a text file
            fRes = fh(self._outputpath+'mcmcResults.txt')
            fRes.writeText('###########################')
            fRes.writeText('####   INFERENCE TEST   ###')
            fRes.writeText('###########################')
            fRes.writeText(' ')
            fRes.writeText('Results of the MCMC fit')
            fRes.writeText('----------------------- ')
            fRes.writeText(' ')
            fRes.writeText('>> Orbit parameter values (highly probable):')
            fRes.writeText(' ')
            for i in range(6):
                confidenceMax = confidenceInterval[self._pKey[i]][1]
                confidenceMin = -confidenceInterval[self._pKey[i]][0]
                if self._pKey[i] == 'longitudeAscendingNode':
                    text = '{}: \t{:.2f} \t\t-{:.2f} \t\t+{:.2f}'
                elif self._pKey[i] == 'periastronTime':            
                    text = '{}: \t\t{:.2f} \t-{:.2f} \t+{:.2f}'
                elif self._pKey[i] == 'eccentricity':
                    text = '{}: \t\t\t{:.4f} \t\t-{:.4f} \t+{:.4f}'
                else:
                    text = '{}: \t\t\t{:.2f} \t\t-{:.2f} \t\t+{:.2f}'
                    
                fRes.writeText(text.format(self._pKey[i],val_max[self._pKey[i]],confidenceMin,confidenceMax))                   
    
        return (val_max,confidenceInterval)


    # ------------------------------------------------------------------------------------------------------------------------------------------                
    def highlyProbableSolution(self, valMax, confidenceInterval, priors, nOrbit = 1000, nSolution = 1, condition = 1, chi2r_all = None, synthetic = False, verbose = False):
        """ Search for the highly probable solution(s) in the Markov chain.
        
        Two methods can be adopted:
        #1 One needs only 1 solution (:param nSolution: set to 1) with a 
        reduced chi2 smaller than a given threshold (:param condition:).
        
        #2 One needs n orbits (:param nOrbit:) with a reduced chi2 smaller 
        than a given threshold.
        """
        # betapicb: np.array([8.61118012e+00,3.57338537e-02,8.90318715e+01,3.16045471e+01,6.60444646e+00,2.44582786e+06]) # chi2r: 0.374606045527

        stat = st()

        if nSolution == 1:
            #parameter = {key: (valMax[key]+confidenceInterval[key][0]) + np.random.rand(nOrbit)*(confidenceInterval[key][1]-confidenceInterval[key][0]) for key in self._pKey}
            
            tt = self._isamples[np.random.choice(range(self._length),nOrbit, replace=False),:]
            if synthetic:
                from Sampler import toKepler
                tt_temp = np.zeros_like(tt)
                for k in range(nOrbit):
                    tt_temp[k,:] = toKepler(tt[k,:],which=priors['which'], mass=priors['starMass'],referenceTime=priors['referenceTime'])
                
                parameter = {self._pKey[j]: tt_temp[:,j] for j in range(6)}
            else:                    
                parameter = {self._pKey[j]: tt[:,j] for j in range(6)}            
            
            chi2Orbit = np.inf
            temp = np.inf
            chi2_all = np.zeros(nOrbit)
            for j in range(nOrbit):
                orbit = o.Orbit(semiMajorAxis = parameter['semiMajorAxis'][j],\
                            eccentricity = parameter['eccentricity'][j],\
                            inclinaison = parameter['inclinaison'][j],\
                            longitudeAscendingNode = parameter['longitudeAscendingNode'][j],\
                            periastron = parameter['periastron'][j],\
                            periastronTime = parameter['periastronTime'][j],\
                            dStar = priors['starDistance'], starMass = priors['starMass'])
     
                #temp = orbit.chiSquare(ob,er)/dof
                temp = stat.chiSquare(orbit,self._ob,self._er)/self._dof
                chi2_all[j] = temp
                if chi2Orbit > temp:
                    chi2Orbit = temp
                    parameterMin = np.array([parameter[key][j] for key in self._pKey])
                    if verbose:
                        print('New chi2r: {}'.format(chi2Orbit)) 
            
            return (parameterMin, chi2Orbit, chi2_all)

        elif nSolution > 1: 
            nOrbit = np.amin([int(1e03*nSolution),self._length])
            
            if chi2r_all is not None:
                n_chi2r = chi2r_all[chi2r_all < condition].shape[0]
                if nSolution > n_chi2r:
                    print("{} solutions requested but there are only {} solution(s) with chi2r <= {}".format(nSolution,n_chi2r,condition))
                    nSolution = n_chi2r
            #parameter = {key: (valMax[key]+confidenceInterval[key][0]) + np.random.rand(nOrbit)*(confidenceInterval[key][1]-confidenceInterval[key][0]) for key in self._pKey}
            
            tt = self._isamples[np.random.choice(range(self._length),nOrbit, replace=False),:]            
            if synthetic:
                from Sampler import toKepler
                tt_temp = np.zeros_like(tt)
                for k in range(nOrbit):
                    tt_temp[k,:] = toKepler(tt[k,:],which=priors['which'], mass=priors['starMass'],referenceTime=priors['referenceTime'])
                
                parameter = {self._pKey[j]: tt_temp[:,j] for j in range(6)}
            else:                    
                parameter = {self._pKey[j]: tt[:,j] for j in range(6)} 

            chi2Orbit = np.ones(nSolution)*np.inf
            temp = np.inf
            k = 0  
            for j in range(nOrbit):
                orbit = o.Orbit(semiMajorAxis = parameter['semiMajorAxis'][j],\
                            eccentricity = parameter['eccentricity'][j],\
                            inclinaison = parameter['inclinaison'][j],\
                            longitudeAscendingNode = parameter['longitudeAscendingNode'][j],\
                            periastron = parameter['periastron'][j],\
                            periastronTime = parameter['periastronTime'][j],\
                            dStar = priors['starDistance'], starMass = priors['starMass'])
     
                temp = stat.chiSquare(orbit,self._ob,self._er)/self._dof
                if temp <= condition:
                    chi2Orbit[k] = temp
                    if k == 0:
                        parameterMin = np.array([parameter[key][j] for key in self._pKey])
                    else:
                        parameterMin = np.vstack((np.array([parameter[key][j] for key in self._pKey]),parameterMin))
                    if verbose:    
                        print('Stored chi2r: {}'.format(chi2Orbit[k]))
                    k += 1
                if k >= nSolution:
                    return (parameterMin, chi2Orbit)
            
            try:
                return (parameterMin, chi2Orbit)
            except:
                return None

    # ------------------------------------------------------------------------------------------------------------------------------------------                
    def showOrbit(self, p, priors, link = True, figsize = (15,15), lim = [], title = None, display = True, save = (False,''), output = False):
        #p = [67.95,0,22.06,55.5,0,2451259.5] # HR8799b
        orbit = o.Orbit(semiMajorAxis=p[0],\
                    eccentricity=p[1],\
                    inclinaison=p[2],\
                    longitudeAscendingNode=p[3],\
                    periastron=p[4],\
                    periastronTime=p[5],\
                    starMass=priors['starMass'],\
                    dStar = priors['starDistance'])
        
        #lim = [[-0.30,-0.10],[-0.42,-0.22]] # betapicb
        #lim = [[1.3,1.7],[0.65,1.05]] # HR8799b
        orbitLine = orbit.showOrbit(unit='arcsec',\
                                addPoints=[self._ob['raPosition'],self._ob['decPosition'],self._er['raError'],self._er['decError']],\
                                lim=lim,\
                                addPosition=self._ob['timePosition'],\
                                link = link,\
                                figsize = figsize,\
                                title = title,\
                                display = display,\
                                save=save,\
                                output = output)    

        return orbitLine

    # ------------------------------------------------------------------------------------------------------------------------------------------                
    def showSolutions(self, valmax, conf, priors, pchi2r = None, pchi2r0 = None, nSolution = 10, condition = 0.6, pointOfView = 'earth', addPosition = [], figsize = (15,15), lim = [], title = None, save = (False,'')):
        # betapicb: np.array([8.61118012e+00,3.57338537e-02,8.90318715e+01,3.16045471e+01,6.60444646e+00,2.44582786e+06]) # chi2r: 0.374606045527        

        if pchi2r is None:        
            p, chi2r = BayesianInference.highlyProbableSolution(self,valmax,conf,priors, nSolution = nSolution, condition = condition)
        else:
            p, chi2r = pchi2r
        
        if pchi2r0 is not None:         
            p0, chi2r0 = pchi2r0 #np.array([8.61118012e+00,3.57338537e-02,8.90318715e+01,3.16045471e+01,6.60444646e+00,2.44582786e+06])
            #chi2r0 = 0.374606045527            
            p = np.vstack((p,p0))
            chi2r = np.hstack((chi2r,chi2r0))
        
            
        if nSolution == 1:
            p = np.array([p])
        indexmin = np.argmin(chi2r)        
        chi2rmin = np.amin(chi2r)
        chi2rmax = np.amax(chi2r)
        chi2diff = chi2rmax-chi2rmin
        
        addL = len(addPosition)
        if addL != 0:
            for new in addPosition:
                self._ob['timePosition'].append(new)
        
        xmin = 0
        ymax = 0        
        
        plt.figure(figsize = figsize)
        plt.hold(True)
        if len(lim) > 0:
            plt.xlim((lim[0][0],lim[0][1]))
            plt.ylim((lim[1][0],lim[1][1]))   
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.axes().set_aspect('equal')
        
        
        if pointOfView == 'earth':
            unit = 'arcsec'
            plt.gca().invert_xaxis() # Then, East is left
        elif pointOfView == 'face-on':
            unit = 'au'
            plt.gca().invert_xaxis() # Then, East is left
        
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
                orbitLine, point = orbit.showOrbit(pointOfView=pointOfView,\
                            unit=unit,\
                            addPoints=[self._ob['raPosition'],self._ob['decPosition'],self._er['raError'],self._er['decError']],\
                            lim=lim,\
                            addPosition=self._ob['timePosition'],\
                            figsize = figsize,\
                            title = title,\
                            output = 'onlyAll')
                coeff = 0.2 + ((chi2rmax-chi2r[j])/chi2diff) * 0.5                    
                pointRotate = np.transpose(np.dot(rotateMatrix,np.transpose(point)))
                orbitRotate = np.transpose(np.dot(rotateMatrix,np.array([orbitLine['raPosition'],orbitLine['decPosition']])))
                #plt.plot(orbitLine['raPosition'],orbitLine['decPosition'],color=(0,97/255.,1), linewidth=1, alpha = coeff)
                plt.plot(orbitRotate[:,0],orbitRotate[:,1],color=(0,97/255.,1), linewidth=1, alpha = coeff)
                plt.plot(pointRotate[:self._nob,0],pointRotate[:self._nob,1],'o', color=(0,97/255.,1), markersize = 2, markeredgecolor = 'none')
                plt.plot(pointRotate[self._nob:,0],pointRotate[self._nob:,1],'o', color=(0/255.,128/255.,0/255.), markersize = 2, markeredgecolor = 'none')
                if np.amin(orbitRotate[:,0]) < xmin:
                    xmin = np.amin(orbitRotate[:,0])
                if np.amax(orbitRotate[:,1]) > ymax:
                    ymax = np.amax(orbitRotate[:,1])                
            else:
                orbitLine, point = orbit.showOrbit(pointOfView=pointOfView,\
                            unit=unit,\
                            addPoints=[self._ob['raPosition'],self._ob['decPosition'],self._er['raError'],self._er['decError']],\
                            lim=lim,\
                            addPosition=self._ob['timePosition'],\
                            figsize = figsize,\
                            title = title,\
                            output = 'onlyAll')
                pointRotate0 = np.transpose(np.dot(rotateMatrix,np.transpose(point)))
                orbitRotate0 = np.transpose(np.dot(rotateMatrix,np.array([orbitLine['raPosition'],orbitLine['decPosition']])))                
                #plt.plot(orbitLine['raPosition'],orbitLine['decPosition'],color=(0,97/255.,1), linewidth=2, alpha = 1)
                                                   
                                                  
                                        
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
            
        plt.plot(orbitRotate0[:,0],orbitRotate0[:,1],color=(1,20/255.,47/255.), linewidth=1.5, alpha = 1)
            
        
            
            
        
    
        if title != None:        
            plt.title(title[0], fontsize=title[1], fontweight='bold')
        
        if save[0]:
            plt.savefig(save[1]+'.pdf')      
        
        return (p,chi2r)