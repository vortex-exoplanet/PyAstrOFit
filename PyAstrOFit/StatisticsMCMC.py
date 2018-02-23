# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
import math
import numpy as np
import random as rand
from astropy import constants as const
from astropy.time import Time
import matplotlib.pyplot as plt
from .KeplerSolver import MarkleyKESolver
import datetime
import os

from . import Orbit as o


class StatisticsMCMC:  # The inheritance is used to access to the trueAnomaly method of the Orbit class.
#class StatisticsMCMC(o.Orbit):
    """ DocString:  
        ----------
        Summary::   
        
        Note::
                                        
    """
    
    p = 6
    pKey = ['semiMajorAxis','eccentricity','inclinaison', 'longitudeAscendingNode', 'periastron', 'periastronTime']
    pKeyAlt = ['u1','u2','u3', 'u4', 'u5', 'u6']


    
    
    def __init__(self): 
        """ The constructor of the class """ 
        # -----------------------
        # Preamble instructions
        # -----------------------

        
        # -----------------------------
        # Main part of the constructor
        # -----------------------------
        #o.Orbit.__init__(self, periastronTime)
        #self.__orbitObject = orbitObject  
        #self.__dStar = dStar
        #self.__starMass = starMass
#        self.publicATTRIBUT = value1            
        
    
    # -----------------------------          
    # Attribut Setters and Getters 
    # -----------------------------          
#    def _get_orbitObject(self):
#        return self.__orbitObject  
#    def _set_orbitObject(self,newParam):
#        self.__orbitObject = newParam    
        
        
    # ------------------------------------------------------------------------------------------------------------------------------------------
    # METHODS
    # ------------------------------------------------------------------------------------------------------------------------------------------
    
#    def method(self, otherParameters): 
#        """ 
#        Summary:: 
#            
#        Arguments::
#            
#        Return::
#                     
#        """    
#        #
#        # INSTRUCTIONS
#        #
#        return outputVariable
# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------   
    def chiSquare(self,orbitObject, obs={}, error={}): # TODO: DocString
        """ Under construction
        Summary::
        
        Arguments::
        
        Return::
        
        """  
        if len(error) == 0:
            error = {'decError' : np.ones(obs['decPosition'].size), 'raError' : np.ones(obs['raPosition'].size)}
            
        timeSequence = obs['timePositionJD']
        #obsPositionSequence = {'decPosition' : obs['decPosition'], 'raPosition' : obs['raPosition']}
        k = 0
        trueAnomalySequence = np.zeros(len(timeSequence))        
        for i in timeSequence:
            trueAnomalySequence[k] = orbitObject.trueAnomaly(i)
            k += 1
            
        modelPositionSequence = orbitObject.positionOnOrbit(trueAnomalySequence,'earth','arcsec')
        
#        chi2Dec =   np.sum(np.power(np.abs(obsPositionSequence['decPosition'] - modelPositionSequence['decPosition']) / error['decError'] , 2)) 
#        chi2Ra =   np.sum(np.power(np.abs(obsPositionSequence['raPosition'] - modelPositionSequence['raPosition']) / error['raError'] , 2)) 
        chi2Dec =   np.sum(np.power(np.abs(obs['decPosition'] - modelPositionSequence['decPosition']) / error['decError'] , 2)) 
        chi2Ra =   np.sum(np.power(np.abs(obs['raPosition'] - modelPositionSequence['raPosition']) / error['raError'] , 2))                     
        #if np.amin([-1,chi2Dec+chi2Ra]) != -1 : # which is True if chi2Dec+chi2Ra is not a number (nan) 
        resu = chi2Dec+chi2Ra        
        if math.isnan(resu):
            return 1e+08 
        else:
            return resu

# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------
    def ctpFunction(self,xPrime=0.,xMu=0.,betaMu=1.): 
        """ Under construction
        Summary:: 
            This method return, for a given xPrime, the value of the candidate 
            transition probability (ctp) function q(x'|x) given by Eq.(12), Ford (2005).            
        Arguments::
            
        Return::
                     
        """ 
        return (1 / np.sqrt(2 * np.pi * np.power(betaMu,2))) * np.exp(-np.power(xPrime-xMu,2)/(2 * np.power(betaMu,2)))
        
# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------
    def candidatFromCTPF(self,xMu=0,betaMu=1): 
        """ Under construction
        Summary:: 
            This method generate a candidate vector of parameters x' from the
            candidate transition probability function q(x'|x) given by Eq.(12), Ford (2005).            
        Arguments::
            
        Return::
                     
        """        
        return rand.gauss(xMu,betaMu)       

# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------   
    def dpDistribution(self, orbitObject, obs={}, error={}): # the desired probability distribution
        """ 
        Summary:: 
            This method return, for a given set \vec{x} of parameters, the value 
            of the so-called desired probability (dp) distribution. 
            
            According to Ford (2005), this distribution is roughly proportional 
            to exp[-chi2(x)/2], for application to radial velocity measurements 
            and orbit determination.
        Arguments::
            
        Return::
                     
        """    
        chi2 = StatisticsMCMC.chiSquare(self,orbitObject, obs, error)  
        #print("chi2",chi2)
        return np.exp(-chi2 / 2.)
        #return -chi2 / 2.

# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------
    def acceptanceProbability(self, orbitObject, orbitObjectPrime, obs={}, error={}): 
        """ Under construction
        Summary:: 
            This method return the value of the so-called acceptance probability
            \alpha(x'|x) (M-H algorithm within the Gibbs Sampler).
        Arguments::
            
        Return::
                     
        """  
        #chi2 = StatisticsMCMC.chiSquare(self,orbitObject, obs, error)
        #chi2Prime = StatisticsMCMC.chiSquare(self,orbitObjectPrime, obs, error)
        
#        return np.minimum(1, np.exp((chi2 - chi2Prime) / 2.))
        f = StatisticsMCMC.dpDistribution(self,orbitObject, obs, error)
        fPrime = StatisticsMCMC.dpDistribution(self,orbitObjectPrime, obs, error) 
        
        #print(chi2,chi2Prime, np.exp(-chi2/2.),np.exp(-chi2Prime/2.),f,fPrime,np.exp((chi2 - chi2Prime) / 2.),fPrime/f,np.amin([1,fPrime/f]))        
        #print("f",f,"f'",fPrime)        
        
        if f == 0:
            if fPrime == 0:
                chi2 = StatisticsMCMC.chiSquare(self,orbitObject, obs, error)
                chi2Prime = StatisticsMCMC.chiSquare(self,orbitObjectPrime, obs, error)
                #print(chi2,chi2Prime)
                if chi2Prime < chi2:
                    return 1
                else:
                    return 0
            else:
                return 1
        else:             
            #return np.minimum(1, fPrime/f)
            return np.amin([1,fPrime/f])


# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------
    def ctpfShow(self, paramTitle = 'No title', mean=0, scaleParameter=1):
        """ Under construction
        Summary:: 
            Show the histogram of 1000 values drawn up from the CTPF distribution.
        Arguments::
            
        Return::
                     
        """         
        N = 100000
        k = 0
        var = np.zeros(N)
        while k < N:
            var[k] = StatisticsMCMC.candidatFromCTPF(self,mean,scaleParameter)
            k += 1
        
        print((mean,scaleParameter))
        #print(var)
        
        xHist, bins = np.histogram(var,40,density=True)
        plt.bar((bins[:-1] + bins[1:])/2, xHist, align='center', width=0.7 * (bins[1]-bins[0]))
        plt.title(paramTitle)


# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------
    def acceptanceRate(self, aCountParam, acceptanceRateInfo):
        """ Under construction
        Summary:: 
            Return the acceptance rate (AR) of a given parameter (whichParam). 
            The AR can be calculated both from: 
            (i) a sample of a given size (acceptanceRateInfo['size']), that is 
            to say from the last acceptanceRateInfo['size'] elements in aCount[whichParam];
            (ii) the entire sample (acceptanceRateInfo['size'] = 'all').
        Arguments::
            
        Return::
                     
        """ 
        aHigh = len(aCountParam) 
        if isinstance(acceptanceRateInfo['size'],str) or aHigh < acceptanceRateInfo['size']: # TODO: one needs to check if this parameter (of type str) is equal to 'all'. Right now, all string will be accepted.
            aLow = 1        
        else:
            aLow = aHigh - acceptanceRateInfo['size']
             
        aIn = np.sum(aCountParam[aLow:aHigh+1]) # +1 because the numpy index [i:j] means "extract from the i-th to the j-th NOT INCLUDED
        aOut = len(aCountParam[aLow:aHigh+1])   # Then, we have add 1 to extract up to the end of the vector
        try:
            result = float(aIn)/float(aOut)
        except ZeroDivisionError:
            result = 0
        
        return result        


# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------
    def phi(self, aRate, acceptanceRateInfo):
        """ Under construction
        Summary:: 
            Return the coefficient \phi required to determine the updated value of the scale parameter.
            It corresponds to the \phi defined in 3.2 (Adaptative step size algorithm) in Ford (2006). 
        Arguments::
                 aRate 
            dict acceptanceRateInfo -
        Return::
                     
        """       
        if aRate > 0.5*acceptanceRateInfo['threshold']:
            return 1.
        elif aRate > 0.2*acceptanceRateInfo['threshold'] and aRate <= 0.5*acceptanceRateInfo['threshold']:
            return 1.5
        elif aRate > 0.1*acceptanceRateInfo['threshold'] and aRate <= 0.2*acceptanceRateInfo['threshold']:
            return 2.
        else: # FIXME: ValueError is raised when one uses the second return line code
            return 1.0
            #return math.log(0.01, aRate / acceptanceRateInfo['threshold']) # therefore, (psi_{mu} / psi_{0})^{phi} = 0.01, which corresponds to the lower limit accepted.
        

# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------
# FIXME: sometimes, the AR of a parameter (e.g. eccentricity) falls to zero and never increases again. One should check the values of the scaleParameter in that case. 
       # Indeed, if the scaleParameter is fallen to zero and the AR is also equal to zero, the latter can't increase again since we have to decrease the scaleParameter, 
       # which is already to zero !!! 
       # However, the scaleParameter shouldn't fall to zero ... so, something wrong happens.

    def acceptanceRateUpdate(self, nu, pKey, acceptanceRateInfo, acceptanceRateAll, aCountAll, scaleParameterAll,scaleParameterInitial, aFreqParam, aSizeParam, optimal, optimalAll, display=True, save=(False,'test',str()), displayInfo=True): #scaleParameterAll2
        """ Under construction
        Summary:: 

        Arguments::

        Return::
        """        
        #print("TEST SCALE PARAMETER UPDATE")
        
        #mcNumber = len(acceptanceRateAll)    
                
        #for nu in range(0,mcNumber):
                        
        for key in pKey:
            acceptanceRateAll[str(nu)][key] = np.append(acceptanceRateAll[str(nu)][key],StatisticsMCMC.acceptanceRate(self,aCountAll[str(nu)][key],acceptanceRateInfo))            
        
        if display or save[0]:
            fig = plt.figure()     
            plt.hold('on')
            color = {pKey[0]: 'b', pKey[1]: 'r', pKey[2]: 'g', pKey[3]: 'y', pKey[4]: 'm', pKey[5]: 'c'}
        
        for key in pKey:
            
            if display or save[0]:                
                plt.plot(acceptanceRateAll[str(nu)][key],color[key])
                plt.ylim([0,1])
                plt.xlabel('step number')
                plt.ylabel('Acceptance rate')        
                plt.suptitle("ACCEPTANCE RATE MONITORING\n", fontsize=14, fontweight='bold')
                plt.figtext(.14,.85,"Markov chain #{}".format(nu))
                plt.figtext(.14,.80,"Scale parameter fine tuning")
                plt.legend(['$a$','$e$','$i$','$\Omega$','$\omega$','$t_\omega$'], fontsize='xx-small')
                        
                
            testValue = np.power(acceptanceRateAll[str(nu)][key][-1] - acceptanceRateInfo['threshold'], 2.)
            threshold = aFreqParam[str(nu)][key] * acceptanceRateInfo['threshold'] * (1 - acceptanceRateInfo['threshold']) / aSizeParam[str(nu)][key]

            if testValue > threshold:                           
                scaleParameterAll[str(nu)][key] = scaleParameterAll[str(nu)][key] * np.power(acceptanceRateAll[str(nu)][key][-1] / acceptanceRateInfo['threshold'], StatisticsMCMC.phi(self,acceptanceRateAll[str(nu)][key][-1], acceptanceRateInfo))
                
                
                if (key == 'inclinaison' or key == 'periastron' or key == 'longitudeAscendingNode') and scaleParameterAll[str(nu)][key] > 4*180:
                    scaleParameterAll[str(nu)][key] = 4*180 
                if scaleParameterAll[str(nu)][key] == 0:
                    scaleParameterAll[str(nu)][key] = scaleParameterInitial[key]

                aSizeParam[str(nu)][key] = acceptanceRateInfo['size']
            else:
                aSizeParam[str(nu)][key] += acceptanceRateInfo['size']
                
#        offset = [acceptanceRateAll[str(nu)][key][-1]-acceptanceRateInfo['threshold'] for key in pKey]
#        stringCurrent = "Current RA: a={:.3f} | e={:.3f} | i={:.3f} | omega={:.3f} | w={:.3f} | tp={:.3f}".format(acceptanceRateAll[str(nu)][pKey[0]][-1],\
#                                                                                                        acceptanceRateAll[str(nu)][pKey[1]][-1],\
#                                                                                                        acceptanceRateAll[str(nu)][pKey[2]][-1],\
#                                                                                                        acceptanceRateAll[str(nu)][pKey[3]][-1],\
#                                                                                                        acceptanceRateAll[str(nu)][pKey[4]][-1],\
#                                                                                                        acceptanceRateAll[str(nu)][pKey[5]][-1])
        #print(stringCurrent)                                                                                                         
                                                                                                        
#        stringOffset = "RA offset: a={:.3f} | e={:.3f} | i={:.3f} | omega={:.3f} | w={:.3f} | tp={:.3f}".format(offset[0],offset[1],offset[2],offset[3],offset[4],offset[5])                                                                                                    
        #print(stringOffset)
        
        
        
        
        k = 30
        boundUP = acceptanceRateInfo['threshold']+0.1
        boundDO = acceptanceRateInfo['threshold']-0.1
        
        for q in range(0,6):
            length = len(acceptanceRateAll[str(nu)][pKey[q]]) 
            start = np.amax([length-k,1])
            s = np.std(acceptanceRateAll[str(nu)][pKey[q]][start:])
            m = np.median(acceptanceRateAll[str(nu)][pKey[q]][start:])
            #fwhm = 2*np.sqrt(2*np.log(2))*s
            #print(pKey[q],m,s,fwhm)  
            if length > k and m < boundUP and m > boundDO and s < 0.20*m:
                optimal[str(nu)][q] = True
                optimalAll[nu] = np.product(optimal[str(nu)])                    
            else:
                optimal[str(nu)][q] = False
                optimalAll[nu] = np.product(optimal[str(nu)])
        stringOptimal = 'MC {}: optimal={}'.format(nu,optimal[str(nu)])
        if displayInfo:
            print(stringOptimal)
        
        if display or save[0]:         
            length = len(acceptanceRateAll['0'][pKey[0]])                        
            plt.plot([0,length],[acceptanceRateInfo['threshold'],acceptanceRateInfo['threshold']],'--k')
            plt.plot([0,length],[boundUP,boundUP],'k')
            plt.plot([0,length],[boundDO,boundDO],'k')  
                
        if save[0]:
            if os.path.exists(save[2]+'scaleParameterPhase/'):
                pass
            else:
                os.makedirs(save[2]+'scaleParameterPhase/')
            plt.savefig(os.path.join(save[2]+'scaleParameterPhase/',save[1]+'_MC'+str(nu)+'.pdf'))
#            f = fh.FileHandler(os.path.join(save[2]+'scaleParameterPhase/',save[1]+'_MC'+str(nu)+'.txt'))
#            f.writeText(stringCurrent)
#            f.writeText(stringOffset)
#            f.writeText(stringOptimal)
#            f.writeText("\n")
                        
        if display:         
            plt.show()    
        else:
            try:
                plt.close(fig)
            except:
                pass

                
# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------

    def algorithmMetropolisHastings(self,n, whichParam, markovChain, scaleParameter, conditions, chiSquare, prior, aCount, candidatMonitoring, obs, error, alternativeCTPF = False):
        """ Under construction
        Summary:: 

        Arguments::

        Return::
        """   

        isPhysical = True
        #         
        # Generate a candidat element x'_\mu for each Markov chain. 
        # 
        candidat = StatisticsMCMC.candidatFromCTPF(self,xMu = markovChain[whichParam][n-1], betaMu = scaleParameter[whichParam]) 
        candidatMonitoring[whichParam] = np.append(candidatMonitoring[whichParam],candidat)

        # TODO: for the case of the candidat is non physical, it would be faster to just reject the candidat and continue to the next step.

        if whichParam == 'inclinaison':
            candidat = np.mod(candidat, conditions['inclinaison'][1])
        if whichParam == 'longitudeAscendingNode':
            candidat = np.mod(candidat, conditions['longitudeAscendingNode'][1])
        if whichParam == 'periastron':
            candidat = np.mod(candidat, conditions['periastron'][1])           
            
        if candidat < conditions[whichParam][0] or candidat > conditions[whichParam][1]: # test if the candidat is physical or not
            isPhysical = False
            

        #
        # Update the n-th step of the Markov chain. The \mu-th element is the candidate and has to be accepted, or not, according to the M.-H. algorithm within the Gibbs sampler.
        # 
        for key in markovChain.keys():
            if key != whichParam: 
                markovChain[key][n] = markovChain[key][n-1]
            else:
                markovChain[key][n] = candidat
                
        if alternativeCTPF:
            try:    
                uParameters = {key: markovChain[key][n] for key in self.pKeyAlt}
                modelParameters = StatisticsMCMC.xFROMu(self,uVector=uParameters)  
            except:
                isPhysical = False
        #
        # M.-H. main part of the algorithm
        #

        if isPhysical:        
            # Generate the orbitObject of the n-th Markov chain step (candidat) and the (n-1)-th one (current state) 
            if len(set(self.pKey)|set(markovChain.keys())) == self.p:
                orbitCandidat = o.Orbit(semiMajorAxis = markovChain['semiMajorAxis'][n], eccentricity = markovChain['eccentricity'][n], inclinaison = markovChain['inclinaison'][n],\
                longitudeAscendingNode = markovChain['longitudeAscendingNode'][n], periastron = markovChain['periastron'][n], periastronTime = markovChain['periastronTime'][n], \
                starMass = prior['starMass'], dStar = prior['dStar'])
            else:
                #uParameters = {key: markovChain[key][n] for key in self.pKeyAlt}
                #modelParameters = StatisticsMCMC.xFROMu(self,uVector=uParameters)
                orbitCandidat = o.Orbit(semiMajorAxis = modelParameters['semiMajorAxis'], eccentricity = modelParameters['eccentricity'], inclinaison = modelParameters['inclinaison'],\
                longitudeAscendingNode = modelParameters['longitudeAscendingNode'], periastron = modelParameters['periastron'], periastronTime = modelParameters['periastronTime'], \
                starMass = prior['starMass'], dStar = prior['dStar'])
        
            chiSquare[n] = StatisticsMCMC.chiSquare(self,orbitCandidat, obs = obs, error = error)
           
            if len(set(self.pKey)|set(markovChain.keys())) == self.p:
                orbitCurrent = o.Orbit(semiMajorAxis = markovChain['semiMajorAxis'][n-1], eccentricity = markovChain['eccentricity'][n-1], inclinaison = markovChain['inclinaison'][n-1],\
                longitudeAscendingNode = markovChain['longitudeAscendingNode'][n-1], periastron = markovChain['periastron'][n-1], periastronTime = markovChain['periastronTime'][n-1], \
                starMass = prior['starMass'], dStar = prior['dStar']) 
            else:
                uParameters = {key: markovChain[key][n-1] for key in self.pKeyAlt}
                modelParameters = StatisticsMCMC.xFROMu(self,uVector=uParameters)
                orbitCurrent = o.Orbit(semiMajorAxis = modelParameters['semiMajorAxis'], eccentricity = modelParameters['eccentricity'], inclinaison = modelParameters['inclinaison'],\
                longitudeAscendingNode = modelParameters['longitudeAscendingNode'], periastron = modelParameters['periastron'], periastronTime = modelParameters['periastronTime'], \
                starMass = prior['starMass'], dStar = prior['dStar'])                
            # TODO: Creating the orbitCurrent object could be avoided. Indeed, one should retrieve it from the previous step:
            # (i)  the previous candidat had been accepted --> the previous orbitCandidat object is still valid for current orbitCurrent object.
            # (ii) the previous candidat had been rejected --> the previous orbitCurrent is still valid for the current orbitCurrent object.
            
            aProb = StatisticsMCMC.acceptanceProbability(self,orbitObject = orbitCurrent, orbitObjectPrime = orbitCandidat, obs = obs, error = error)
        else:
            chiSquare[n] = 1e+12
            aProb = 0.    
        
        u = rand.uniform(0,1)

        if u > aProb: # In this case, the candidat is rejected. One needs to update the mu-th element of the n-th step of the Markov chain, back to the (n-1)-th step value.
            markovChain[whichParam][n] = markovChain[whichParam][n-1]
            aCount[whichParam] = np.append(aCount[whichParam],0)
            #aCountAll[whichParam] = np.append(aCountAll[whichParam],0)
        else: # the candidat is accepted, and we only have to update the aCount
            aCount[whichParam] = np.append(aCount[whichParam],1)
            #aCountAll[whichParam] = np.append(aCountAll[whichParam],1)
    
        #return markovChain, chiSquare, aCount

# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------        
    def uFROMx(self,param , referenceTime, mass):               
        """ Under construction
        Summary:: 
            Return the vector u(x), Eq.(A.1) from Chauvin et al. (2012), from a given vector x = (a, e, i, Omega, w, tp).
        Arguments::
            
        Return::
                     
        """ 
        #param = np.array([8.8,0.021,88.5,45,78,2453846.3])
        
        if param[2] == 0.0:
            param[2] = 1e-12
        
        #w = math.radians(param['periastron'])
        #omega = math.radians(param['longitudeAscendingNode'])
        #i = math.radians(param['inclinaison'])        
        period = np.sqrt(4*np.power(np.pi,2)*np.power(param[0]*const.au.value,3) / (const.G.value * mass*const.M_sun.value)) / (3600.*24.*365) # given in years (365 days/year, 24hrs/day)
        
                
        # TODO: The calculus of the mean anomaly should be performed from the method 'trueAnomaly' defined in the class Orbit. Therefore, one needs to use the inheritance.
        if isinstance(referenceTime,str): 
            referenceTime = Time(referenceTime,format='iso',scale='utc').jd 
            
        if isinstance(param[5],str): 
            param[5] = Time(param[5],format='iso',scale='utc').jd    
            
        meanAnomaly = np.mod(2*np.pi * ((referenceTime-param[5])/365) / period, 2*np.pi)
        if param[1] <= 0.05: # then we use an approximate expression of E as a function of  M and e
            eccentricAnomaly = meanAnomaly + (param[1] - np.power(param[1],3)/8) * math.sin(meanAnomaly) + np.power(param[1],2)/2 * math.sin(2 * meanAnomaly) + 3*np.power(param[1],3)/8 * math.sin(3 * meanAnomaly)
        else: # otherwise, we use the Markley algorithm
            ks = MarkleyKESolver() # call the Kepler equation solver
            eccentricAnomaly = ks.getE(meanAnomaly,param[1])
        
        temp = 2 * math.atan(math.sqrt((1+param[1])/(1-param[1])) * math.tan(eccentricAnomaly / 2)) # TODO il faut tester l'unicité de la solution
        nu0 = np.mod(temp,2*np.pi)  
        #print('nu0 vaut',math.degrees(nu0))               
        
        u1 = np.cos(math.radians(param[4]) + math.radians(param[3]) + nu0) / period 
        u2 = np.sin(math.radians(param[4]) + math.radians(param[3]) + nu0) / period 
        
        u3 = param[1] / np.sqrt(1 - np.power(param[1],2)) * np.cos(math.radians(param[4]) + math.radians(param[3]))
        u4 = param[1] / np.sqrt(1 - np.power(param[1],2)) * np.sin(math.radians(param[4]) + math.radians(param[3]))
        
        u5 = np.power(1 - np.power(param[1],2),0.25) * np.sin(0.5 * math.radians(param[2])) * np.cos(math.radians(param[4]) - math.radians(param[3]))
        u6 = np.power(1 - np.power(param[1],2),0.25) * np.sin(0.5 * math.radians(param[2])) * np.sin(math.radians(param[4]) - math.radians(param[3]))
        
        #uVector =  {'u1' : u1, 'u2' : u2, 'u3' : u3, 'u4' : u4, 'u5' : u5, 'u6' : u6}
        return np.array([u1,u2,u3,u4,u5,u6])
        
# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------    
    def xFROMu(self, uVector, referenceTime, mass):
        """ Under construction
        Summary:: 
            Return the vector x(u) by inverting the Eq.(A.1) from Chauvin et al. (2012).
        Arguments::
            
        Return::
                     
        """     

        if uVector[4] == 0 and uVector[5] == 0:
            uVector[4] = 1e-15      
            uVector[5] = 1e-15
        
        
        #a = np.power(uVector[0],2) + np.power(uVector[1],2)
        a = uVector[0]**2 + uVector[1]**2
        b = np.power(uVector[2],2) + np.power(uVector[3],2)
        c = np.power(uVector[4],2) + np.power(uVector[5],2) 
                
        period = np.sqrt(1./a) # given in years 
        eccentricity = np.sqrt(b / (1 + b))
        
        semiMajorAxis = np.power(const.G.value * mass*const.M_sun.value * np.power(period*365*24*3600,2.)/ (4 * np.power(np.pi,2.)),1./3.) # semiMajorAxis given in meter.
        
        fct_bc = np.sqrt(c * np.sqrt(1+b))
        if np.abs(fct_bc) >= 1:
            #print('NON PHYSICAL')
            return np.zeros(6)
        inclinaison = (math.degrees(2*np.arcsin(fct_bc)) , math.degrees(2*np.pi - 2*np.arcsin(fct_bc)))
        #print c, b
        inclinaison = np.amin(inclinaison)       
                
        #print semiMajorAxis/const.au.value, eccentricity, inclinaison
        
        k3 = eccentricity / np.sqrt(1 - np.power(eccentricity,2))
        k5 = np.power(1 - np.power(eccentricity,2),0.25) * np.sin(0.5 * math.radians(inclinaison))
        
        qi1 = k5 * uVector[2] + k3 * uVector[4]
        qi2 = k5 * uVector[3] + k3 * uVector[5]
        qi3 = k5 * uVector[3] - k3 * uVector[5]
        #qi4 = k3 * uVector['u5'] - k5 * uVector['u3']
        #print("q1, q2, q3",qi1,qi2,qi3)
        
        wStar = np.mod(np.arctan2(qi2,qi1),2*np.pi)
        omegaStar = np.mod(np.arctan2(qi3,qi1),2*np.pi)
        #print("wStar, omegaStar",wStar,omegaStar)
        
        if wStar < np.pi/2. and omegaStar < np.pi/2.:
            case = 1
            wADDo = np.mod(wStar+omegaStar,2*np.pi)
            w1 = wStar
            omega1 = omegaStar
            w2 = wStar + np.pi
            omega2 = omegaStar + np.pi
        elif wStar < np.pi/2. and omegaStar > 3*np.pi/2.:
            case = 2
            wADDo = np.mod(wStar+omegaStar,2*np.pi)
            w1 = wStar
            omega1 = omegaStar
            w2 = wStar + np.pi
            omega2 = omegaStar - np.pi
        elif np.mod(wStar-np.pi/2.,2*np.pi) < np.pi/2. and np.mod(omegaStar-np.pi/2.,2*np.pi) < np.pi/2.:
            case = 3
            wADDo = np.mod(wStar+omegaStar+np.pi,2*np.pi)
            w1 = wStar
            omega1 = omegaStar + np.pi
            w2 = wStar + np.pi
            omega2 = omegaStar
        elif np.mod(wStar-np.pi/2.,2*np.pi) < np.pi/2. and np.mod(omegaStar-np.pi,2*np.pi) < np.pi/2.:
            case = 4
            wADDo = np.mod(wStar+omegaStar+np.pi,2*np.pi)
            w1 = wStar
            omega1 = omegaStar - np.pi
            w2 = wStar + np.pi
            omega2 = omegaStar
        elif np.mod(wStar-np.pi,2*np.pi) < np.pi/2. and np.mod(omegaStar-np.pi/2.,2*np.pi) < np.pi/2.:
            case = 5
            wADDo = np.mod(wStar+omegaStar-np.pi,2*np.pi)
            w1 = wStar
            omega1 = omegaStar + np.pi
            w2 = wStar - np.pi
            omega2 = omegaStar
        elif np.mod(wStar-np.pi,2*np.pi) < np.pi/2. and np.mod(omegaStar-np.pi,2*np.pi) < np.pi/2.:
            case = 6
            wADDo = np.mod(wStar+omegaStar-np.pi,2*np.pi)
            w1 = wStar
            omega1 = omegaStar - np.pi
            w2 = wStar - np.pi
            omega2 = omegaStar
        elif np.mod(wStar-3*np.pi/2.,2*np.pi) < np.pi/2. and np.mod(omegaStar,2*np.pi) < np.pi/2.:
            case = 7
            wADDo = np.mod(wStar+omegaStar,2*np.pi)
            w1 = wStar
            omega1 = omegaStar
            w2 = wStar - np.pi
            omega2 = omegaStar + np.pi
        elif wStar > 3*np.pi/2. and omegaStar > 3*np.pi/2.:
            case = 8
            wADDo = np.mod(wStar+omegaStar,2*np.pi)
            w1 = wStar
            omega1 = omegaStar
            w2 = wStar - np.pi
            omega2 = omegaStar - np.pi
        else:
            print((wStar,omegaStar))
            print("J'ai merdé quelque part ...")
            print(stop)
            #print(uVector)
            #w1, w2, omega1, omega2, wADDo = (0,0,0,0,0)
                  
        w1 = np.mod(w1,2*np.pi) 
        w2 = np.mod(w2,2*np.pi) 
        omega1 = np.mod(omega1,2*np.pi)
        omega2 = np.mod(omega2,2*np.pi) 
                
        ####### periastronTime
        nu0 = np.mod(np.arctan2(uVector[1],uVector[0])-wADDo,2*np.pi)
        eccAnomaly = 2*np.arctan(np.tan(nu0/2.) / (np.sqrt((1+eccentricity)/(1-eccentricity))))
        meanAnomaly = np.mod(eccAnomaly - eccentricity*np.sin(eccAnomaly),2*np.pi)
        if isinstance(referenceTime,str): 
            referenceTime = Time(referenceTime,format='iso',scale='utc').jd    
        tp = referenceTime - meanAnomaly/(2*np.pi)*(period*365)
        
        # we want to return the non passed periastronTime which is the closest to the current time
        # TODO: we have to implement the case where nowjd-tp < 0, that is whether the tp is in the future.
        now = datetime.datetime.now().isoformat()
        now = now[:10] + ' ' + now[11:len(now)]
        nowjd = Time(now,format='iso',scale='utc').jd  
        quotient = (referenceTime-tp) // (365*period)        
        tp = tp + quotient*1*period


#        try:        
#            periastronTime = Time(tp,format='jd',scale='utc').iso
#        except:
#            periastronTime = "0001-01-01 00:00:00.0"
        ########
                
#        print("\n")
#        print(" case {}".format(case))
#        print(math.degrees(omegaStar),math.degrees(wStar))
#        print('omega1 w1',math.degrees(omega1),math.degrees(w1))
#        print('omega2 w2',math.degrees(omega2),math.degrees(w2))        
#        print("\n")
        
        
        #xVector = {'semiMajorAxis' : semiMajorAxis/const.au.value, 'eccentricity' : eccentricity, 'inclinaison' : inclinaison, 'periastron' : math.degrees(w1), 'longitudeAscendingNode' : math.degrees(omega1), 'periastronTime' : periastronTime, 'period' : period}
        return np.array([semiMajorAxis/const.au.value,eccentricity,inclinaison,math.degrees(omega1),math.degrees(w1),tp])
        
        
        
        
        
        