# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 07:45:24 2014

@author: Dr. Olivier Wertz
"""

__all__ = ["Orbit"]

import math
import numpy as np

from astropy import constants as const
from astropy.time import Time
import matplotlib.pyplot as plt

from Toolbox import timeConverter
from KeplerSolver import MarkleyKESolver


class Orbit:
    """ 
    Create a numerical model of a planet orbit. 

    Parameters
    -----------                       

    semiMajorAxis: float
        The semi-major axis in AU.
    
    eccentricity: float (0 <= e < 1)
    
    inclinaison: float
        The inclination in deg.
    
    longitudeAscendingNode: float
        The longitude of the ascending node in deg.
    
    periastron: float
        The argument of periastron in deg.
    
    periastronTime: float or str
        The time for periastron passage. This parameter can be given in iso
        format (str) or directly in JD (float)

    dStar: float
        Distance between the star and Earth in pc.
        
    starMass: float
        The mass of the star given in solar masses.
        
    period: float
         The orbital period of the planet in year.

    Notes
    -----                        
    Let us note that a, p and mStar are linked by the 3rd Kepler law:
    a^3 / P^2 = G (mStar + mPlanet) / (4 pi^2) .
    Since we neglect mPlanet in comparision with mStar, only two 
    parameters (among a, p and mStar) need to be defined.
                                        
    """
    
    def __init__(self,semiMajorAxis = 0, period = 0, starMass = 0, eccentricity = 0, inclinaison = 0, longitudeAscendingNode = 0, periastron = 0, periastronTime = "2000-01-01 00:00:00.0", dStar = 1): #add planetTypeParameters in order to initialize the parameters according to a particular family of planet
        """ The constructor of the class """
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Depending on which parameters (among semiMajorAxis, period and starMass) the user has initialized, we fix or calculate the others from the 3rd Kepler law
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        if semiMajorAxis == 0 and period == 0 and starMass == 0: # No parameters initialized by the user ==> Earth orbit parameters are chosen            
            starMass, period, semiMajorAxis = (1.,1.,1.)
        elif semiMajorAxis != 0:            
            if period == 0 and starMass == 0: # only semiMajorAxis has been initialized by the user ==> we fix period and calculate starMass
                period = 1
                starMassKg = (np.power(semiMajorAxis * const.au.value,3) / np.power(period * (365.*24*3600),2)) * 4 * np.power(np.pi,2) / const.G.value
                starMass = starMassKg / const.M_sun.value
            elif period == 0 and starMass != 0: #  semiMajorAxis and starMass have been initialized by the user ==> we calculate period
                periodSecSquared = (np.power(semiMajorAxis * const.au.value,3) / (starMass * const.M_sun.value)) * 4 * np.power(np.pi,2) / const.G.value
                period = np.power(periodSecSquared,1./2.) / (365.*24*3600)
            elif period != 0 and starMass == 0: #  semiMajorAxis and period have been initialized by the user ==> we calculate starMass
                starMassKg = (np.power(semiMajorAxis * const.au.value,3) / np.power(period * (365.*24*3600),2)) * 4 * np.power(np.pi,2) / const.G.value
                starMass = starMassKg / const.M_sun.value
            else: # all parameters have been initialized by the user. One need to check the physical consistency of the parameters 
                aCube = np.power(period * (365.*24*3600),2) * const.G.value / (4 * np.power(np.pi,2)) * (starMass * const.M_sun.value)
                aTest = np.power(aCube,1./3.) / const.au.value # in AU
                comp = np.abs(aTest - semiMajorAxis) / aTest * 100
                if comp > 3.5:
                    print("Warning:: the values given for the semiMajorAxis, period and starMass are not consistent with")
                    print("the 3rd Kepler law (error larger than {0}%). You might reconsider these values".format(int(np.floor(comp*10))/10.))                                                    
        elif period != 0:
            if semiMajorAxis == 0 and starMass == 0: # only period has been initialized by the user ==> we fix semiMajorAxis and calculate starMass
                semiMajorAxis = 0
                starMassKg = (np.power(semiMajorAxis * const.au.value,3) / np.power(period * (365.*24*3600),2)) * 4 * np.power(np.pi,2) / const.G.value
                starMass = starMassKg / const.M_sun.value
            elif semiMajorAxis == 0 and starMass != 0: #  period and starMass have been initialized by the user ==> we calculate semiMajorAxis
                semiMajorAxisMCube = np.power(period * (365.*24*3600),2) * const.G.value / (4 * np.power(np.pi,2)) * (starMass * const.M_sun.value)
                semiMajorAxis= np.power(semiMajorAxisMCube,1./3.) / const.au.value # in AU                                 
        elif starMass != 0: # only starMass has been initialized by the user ==> we fix period and calculate semiMajorAxis
            period = 1
            semiMajorAxisMCube = np.power(period * (365.*24*3600),2) * const.G.value / (4 * np.power(np.pi,2)) * (starMass * const.M_sun.value)
            semiMajorAxis= np.power(semiMajorAxisMCube,1./3.) / const.au.value # in AU     
        else:
            print("Did I forget a case ?")
        
        
        if isinstance(periastronTime, str):
            periastronTime = Time(periastronTime,format='iso',scale='utc').jd
        # -----------------------------
        # Main part of the constructor
        # -----------------------------
        self.daysInOneYear = 365. #number of days of 24hrs in one year (earth's orbit: 365.256363004 days of 24hrs)           
           
        self.__semiMajorAxis = semiMajorAxis
        self.__period = period
        self.__starMass = starMass
        self.__eccentricity = eccentricity
        self.__inclinaison = math.radians(inclinaison)
        self.__longitudeAscendingNode = math.radians(longitudeAscendingNode)
        self.__periastron = math.radians(periastron)
        self.dStar = dStar  
        self.__periastronTime = periastronTime
        
    
    # -----------------------------          
    # Attribut Setters and Getters 
    # -----------------------------
    # TODO: property !!!!          
#    def _get_semiMajorAxis(self):
#        return self.__semiMajorAxis   
#    def _set_semiMajorAxis(self,newParam):
#        self.__semiMajorAxis = newParam    
#    def _get_period(self):
#        return self.__period   
#    def _set_period(self,newParam):
#        self.__period = newParam   
#    def _get_starMass(self):
#        return self.__starMass  
#    def _set_starMass(self,newParam):
#        self.__starMass = newParam        
#    def _get_eccentricity(self):
#        return self.__eccentricity   
#    def _set_eccentricity(self,newParam):
#        self.__eccentricity = newParam        
#    def _get_inclinaison(self):
#        return math.degrees(self.__inclinaison)   
#    def _set_inclinaison(self,newParam):
#        self.__inclinaison = math.radians(newParam)      
#    def _get_longitudeAscendingNode(self):
#        return math.degrees(self.__longitudeAscendingNode)   
#    def _set_longitudeAscendingNode(self,newParam):
#        self.__longitudeAscendingNode = math.radians(newParam)
#    def _get_periastron(self):
#        return math.degrees(self.__periastron)   
#    def _set_periastron(self,newParam):
#        self.__periastron = math.radians(newParam) 
#    def _get_periastronTime(self):
#        return self.__periastronTime   
#    def _set_periastronTime(self,newParam):
#        self.__periastronTime = Time(newParam,format='iso',scale='utc').jd         
        
    # --------
    # Methods
    # --------
# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------     
    def whichParameters(self):  #TODO : add periastronTime, ...
        """ 
        Summary:: 
            Return a dict-type object which contains the orbital parameters
        Arguments::
            None
        Return::
            dict-type         
        """        
        return {"semiMajorAxis" : self.__semiMajorAxis, "period" : self.__period, "starMass" : self.__starMass,\
        "eccentricity" : self.__eccentricity, "inclinaison" : self.__inclinaison, "longitudeAscendingNode" : self.__longitudeAscendingNode,\
        "periastron" : self.__periastron, "periatronTime" : self.__periastronTime}

# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------         
    def positionOnOrbit(self,trueAnomaly=0,pointOfView="earth",unit="au"):
        """
        Summary::
            Calculation of the position x and y of the planet from the use of the
            orbital parameters. The position of the star is (0,0)
        Arguments::
            - trueAnomaly: value(s) of the true anomaly in degree (default 0). Multi-dimension requires np.array
            - pointOfView: representation of the orbit viewed from the 'earth' (default) 
              or 'face-on'
            - unit: x and y positions given in 'au' (default) or 'arcsec'
        Return::
            dict-type variable {"decPosition": |ndarray, size->pointNumber| , "raPosition" : |ndarray, size->pointNumber}        
        """

        try:
            if pointOfView.lower() == "earth":
                pov = 1
            elif pointOfView.lower() == "face-on":
                pov = 0
            else:     
                print("The pointOfView argument must be 'earth' or 'face-on'.")
                print("As a consequence, the default value has been chosen => 'earth'")
                pov = 1
        except AttributeError:
            print("The pointOfView argument requires a string-type input ('earth' or 'face-on')")
            print("As a consequence, the default value has been chosen => 'earth'")
            pov = 1
            
        try:
            if unit.lower() == "au":
                uni = 1
            elif unit.lower() == "arcsec":
                uni = 0
            else:     
                print("The unit argument must be 'au' or 'arcsec'.")
                print("As a consequence, the default value has been chosen => 'au'")
                uni = 1
        except AttributeError:
            print("The unit argument requires a string-type input ('au' or 'arcsec')")
            print("As a consequence, the default value has been chosen => 'au'")
            uni = 1            
                    
        
        v = trueAnomaly
        
        r = self.__semiMajorAxis * (1 - np.power(self.__eccentricity,2)) / (1 + self.__eccentricity * np.cos(v))

        if uni == 0: # r will be converted in arcsec thanks to the earth - star distance (dStar). 
                     # The latter, which is initially expressed in pc, is converted in au. 
                     # Finally, r is converted in arcsec        
            r = (r / (self.dStar * const.pc.value / const.au.value)) * (180 / np.pi) * 3600
   
        x = r * (np.cos(self.__periastron + v) * np.cos(self.__longitudeAscendingNode) - np.sin(self.__periastron + v) * np.cos(self.__inclinaison * pov) * np.sin(self.__longitudeAscendingNode))
        y = r * (np.cos(self.__periastron + v) * np.sin(self.__longitudeAscendingNode) + np.sin(self.__periastron + v) * np.cos(self.__inclinaison * pov) * np.cos(self.__longitudeAscendingNode))
        x = np.round(x,7) # We only need 7 floating point numbers
        y = np.round(y,7) #
        return {"decPosition" : x, "raPosition" : y}        

# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------     
    def fullOrbit(self,pointNumber=100,pointOfView="earth",unit="au"):
        """
        Summary::
            Calculation of the full orbit of the planet based on the 
            orbital parameters. The position of the star is (0,0)
        Arguments::
            - pointNumber: number of points which compose the orbit (default 100)
            - pointOfView: representation of the orbit viewed from the 'earth' (default) 
              or 'face-on'
            - unit: x and y positions given in 'au' (default) or 'arcsec'
        Return::
            dict-type variable {"decPosition": |ndarray, size->pointNumber| , "raPosition" : |ndarray, size->pointNumber}        
        """
        return Orbit.positionOnOrbit(self,np.linspace(0,2*np.pi,pointNumber),pointOfView,unit)

# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------                     
    def showOrbit(self,pointOfView="earth",unit="au",addPosition=[], addPoints=[], 
                  lim=None, show=True, link=False, figsize=(10,10), title=None, 
                  display=True, save=(False,str()), output=False, 
                  addPosition_options={}, invert_xaxis=True, invert_yaxis=False,
                  cardinal=(False,''), **kwargs):
        """ 
        Under construction 
        
        """  
        if isinstance(addPosition,list) == False:
            raise ValueError("The type of the parameter 'addPosition' is {}. It should be a list containing str or float/int.".format(type(addPosition)))
            
        
        o = Orbit.fullOrbit(self,300,pointOfView,unit)
        if output == 'only' or output == 'onlyAll':
            k = 0
            if output == 'onlyAll':
                for pos in addPosition:                    
                    if type(pos) == str:
                        pos = timeConverter(time=pos)
                        ta1 = Orbit.trueAnomaly(self,pos)
                    elif (type(pos) == float or type (pos) == int or type(pos) == np.float64) and pos > 360: # it this case, the pos value should correspond to the JD time associated to a position 
                        ta1 = Orbit.trueAnomaly(self,pos)
                    else: # we consider pos as the value of the true anomaly (already in radian)                
                        ta1 = pos
                        
                    pointtemp = Orbit.positionOnOrbit(self,ta1,pointOfView,unit)
                    if k == 0:    
                        point = np.array([pointtemp['raPosition'],pointtemp['decPosition']])                    
                    elif k != 0:
                        point = np.vstack((point,np.array([pointtemp['raPosition'],pointtemp['decPosition']])))
                    k += 1    
                return (o,point)
            else:
                return o
        p = Orbit.positionOnOrbit(self,0,pointOfView,unit) # position of the periapsis
        an = Orbit.positionOnOrbit(self,-self.__periastron + np.array([0,np.pi]),pointOfView,unit) # position of the ascending node         
        
        ############
        ## FIGURE ##
        ############
        if lim is None:
            lim1 = np.floor(13.*np.max(np.abs(o["decPosition"]))) / 10. 
            lim2 = np.floor(13.*np.max(np.abs(o["raPosition"]))) / 10.
            lim = np.max([lim1,lim2])
            lim = [(-lim,lim),(-lim,lim)]        
        
        plt.figure(figsize = figsize)
        plt.hold(True)
        plt.plot(0,0,'ko') # plot the star
        if show:            
            orbit_color = kwargs.pop('color',(1-0/255.,1-97/255.,1-255/255.))
            plt.plot(o["raPosition"],o["decPosition"], color=orbit_color, linewidth=2) # plot the orbit        
            plt.plot(p["raPosition"],p["decPosition"],'gs') # plot the position of the periapsis
            plt.plot(an["raPosition"],an["decPosition"],"k--")
        
        plt.xlim((lim[0][0],lim[0][1]))
        plt.ylim((lim[1][0],lim[1][1]))   
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.axes().set_aspect('equal')
        
        # Ticks
#        xticks_bounds = [np.ceil(lim[0][0]*10),np.floor(lim[0][1]*10)]
#        xticks = np.arange(xticks_bounds[0]/10.,xticks_bounds[1]/10.,0.1)
#        plt.xticks(xticks,xticks,size=20)
#        
#        yticks_bounds = [np.ceil(lim[1][0]*10),np.floor(lim[1][1]*10)]
#        yticks = np.arange(np.floor(yticks_bounds[0]/10.),np.floor(yticks_bounds[1]/10.),0.1)
#        plt.yticks(yticks,yticks,size=20)    
        
        
        # Title
        if title != None:           
            plt.title(title[0], fontsize=title[1])#, fontweight='bold')
        
        # Unit
        if unit == 'au':
            plt.xlabel("x' (AU)")
            plt.ylabel("y' (AU)")
        else:                
            plt.xlabel("RA ({0})".format(unit), fontsize=20)
            plt.ylabel("DEC ({0})".format(unit), fontsize=20)
        
        pointTemp = [np.zeros([len(addPosition)]) for j in range(2)];
        
        # Add position 
        addPosition_marker = addPosition_options.pop('marker','s')
        addPosition_markerfc = addPosition_options.pop('markerfacecolor','g')
        for k, pos in enumerate(addPosition): # for all the element in the list-type addPosition, we check the type of the element. 
            if type(pos) == str: # if it's a str which give a date, we need to call trueAnomaly
                pos = timeConverter(time=pos)
            
                ta1 = Orbit.trueAnomaly(self,pos)
            elif (type(pos) == float or type (pos) == int or type(pos) == np.float64) and pos > 360: # it this case, the pos value should correspond to the JD time associated to a position 
                ta1 = Orbit.trueAnomaly(self,pos)
            else: # we consider pos as the value of the true anomaly (already in radian)                
                ta1 = pos
                
            point = Orbit.positionOnOrbit(self,ta1,pointOfView,unit)

            plt.plot(point["raPosition"],
                     point["decPosition"],
                     marker=addPosition_marker,
                     markerfacecolor=addPosition_markerfc,
                     **addPosition_options)
            
            pointTemp[0][k] = point["raPosition"]
            pointTemp[1][k] = point["decPosition"]
        
        # Add points
        if addPoints and len(addPoints) == 2: # test whether the list is empty using the implicit booleanness of the empty sequence (empty sequence are False) 
            plt.plot(addPoints[0], addPoints[1],'b+') 
        elif addPoints and len(addPoints) == 4:
            plt.plot(addPoints[0], addPoints[1],'ko')
            plt.errorbar(addPoints[0],addPoints[1], xerr = addPoints[2], yerr = addPoints[3], fmt = 'k.')
        if link:
            link_color = kwargs.pop('link_color','r')
            for j in range(len(pointTemp[0])):
                plt.plot([pointTemp[0][j],addPoints[0][j]],
                         [pointTemp[1][j],addPoints[1][j]],
                         color=link_color)
        # Axis invert                     
        if invert_xaxis:
            plt.gca().invert_xaxis()
        
        if invert_yaxis:
            plt.gca().invert_yaxis()
            
        # Cardinal 
        if cardinal[0]:            
            x_length = (lim[0][1] - lim[0][0])/8.
            y_length = x_length
            
            if cardinal[1] == 'br':
                cardinal_center = [lim[0][0],lim[1][0]] # bottom right
            elif cardinal[1] == 'bl':
                cardinal_center = [lim[0][1]-1.5*x_length,lim[1][0]]
            
            plt.plot([cardinal_center[0]+x_length/4.,cardinal_center[0]+x_length],
                     [cardinal_center[1]+y_length/4.,cardinal_center[1]+y_length/4.], 
                     'k',
                     linewidth=1)
    
            plt.plot([cardinal_center[0]+x_length/4.,cardinal_center[0]+x_length/4.],
                     [cardinal_center[1]+y_length/4.,cardinal_center[1]+y_length], 
                     'k',
                     linewidth=1)    
            
            plt.text(cardinal_center[0]+x_length+x_length/3.,
                     cardinal_center[1]+y_length/4.-y_length/12,
                     '$E$',
                     fontsize=20)        
    
            plt.text(cardinal_center[0]+x_length/4.+x_length/8,
                     cardinal_center[1]+y_length+y_length/6.,
                     '$N$',
                     fontsize=20)                 

        # Save
        if save[0]:
            plt.savefig(save[1]+'.pdf')
        if display:    
            plt.show()   
        
        if output:
            return o
# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------     
    def trueAnomaly(self,time): # TODO :: 1) tenir compte du format de time et self_periastronTime. 2) prendre en charge les array de time
        """ Under construction 
        Summary::
            Return the true anomaly of the planet at a given time according to 
            the orbit parameters and the time for passage to periastron
        Arguments::
            time: observation time at which we want to calculate the true anomaly. 
                  The format can be either ISO (e.g. '2000-01-01 00:00:00.0') or JD
                  (e.g. 2451544.5)
        
        Return::
        
        """       
        if isinstance(time,float) == False and isinstance(time,int) == False: 
            time = Time(time,format='iso',scale='utc').jd # conversion in JD 

        meanAnomaly = (2*np.pi * ((time-self.__periastronTime)/self.daysInOneYear) / self.__period) % (2*np.pi)
        ks = MarkleyKESolver() # call the Kepler equation solver
        eccentricAnomaly = ks.getE(meanAnomaly,self.__eccentricity)  

        mA1 = 2 * math.atan(math.sqrt((1+self.__eccentricity)/(1-self.__eccentricity)) * math.tan(eccentricAnomaly / 2)) # TODO il faut tester l'unicité de la solution
        return mA1 % (2*np.pi)                 



# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------     
    def eccentricAnomaly(self,x): # TODO :: 1) tenir compte du format de time et self_periastronTime. 2) prendre en charge les array de time
        """ Under construction 
        Summary::

        Arguments::
        
        Return::
        
        """
        if x == 0:
            return 0
        elif x > 1:
            x = x-2
            wrap = True
        else:
            wrap = False
                            
        beta = 7*np.pi**2/60.
        gamma = 7./(3.*beta)*(1-2*(beta-1)*(1+self.__eccentricity)) - 1

        x2 = 3./4.-self.__eccentricity/(2**(1./2.)*np.pi)
        F = 1/(self.__eccentricity * (1 - (3./4.)**2 * (beta -(beta - 1)*x2**2)) / (1 + (3./7.)*beta*(3./4.)**2 * (1+gamma*x2**2)))
        alpha = x2**(-4)*(24/np.pi**2*(1-self.__eccentricity)**3+self.__eccentricity*x2**2)*(1+x2**2)/(1-x2**2)*((1-4*x2/3.)*F-1)
        fx =  1 + alpha * x**4/(24/np.pi**2*(1-self.__eccentricity)**3+self.__eccentricity*x**2) *(1-x**2)/(1+x**2)
        
        f0 = -x
        f1 = 1 - self.__eccentricity*fx
        f2 = -(3./7.)*beta*x*(1+gamma*x**2)
        f3 = self.__eccentricity*fx*(beta-(beta-1)*x**2) + (3./7.)*beta*(1+gamma*x**2)
        
        p = (1./3.) * (f1/f3 - (1./3.)*(f2/f3)**2)
        q = (1./2.) * ((1./3.)*f2*f1/f3**2 - f0/f3 - (2./27.)*(f2/f3)**3)

        cp = 1 + (p**3/q**2 + 1)**(1./2.)
        cm = 1 - (p**3/q**2 + 1)**(1./2.)

        if q >= 0:
            qsigne = 1            
        else:
            qsigne = -1
            q = -q
        if cm >= 0:
            cmsigne = 1
        else:
            cmsigne = -1 
            cm = -cm
        if cp >= 0:
            cpsigne = 1
        else:
            cpsigne = -1  
            cp = -cp
        temp = qsigne*q**(1./3.) * (cpsigne*cp**(1./3.) + cmsigne*cm**(1./3.))
        
        y = -(1./3.) * f2/f3 + temp
        
        if wrap:
            return 2+y
        else:
            return y
# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------  
    def timeFromTrueAnomaly(self, givenTrueAnomaly, output = 'iso'):
        """ Under construction 
        Summary::
        
        Arguments::
        
        Return::
        
        """
        if givenTrueAnomaly > 2*np.pi:
            givenTrueAnomaly = math.radians(givenTrueAnomaly)

            
        cosE = (self.__eccentricity+np.cos(givenTrueAnomaly))/(self.__eccentricity*np.cos(givenTrueAnomaly)+1)         
        sinE = np.sin(givenTrueAnomaly)/np.sqrt(1-self.__eccentricity**2)*(1-self.__eccentricity*cosE)
        
        eccAnomaly =2*np.arctan(np.tan(givenTrueAnomaly/2.)/np.sqrt((1+self.__eccentricity)/(1-self.__eccentricity)))
        
        if np.sin(eccAnomaly) != sinE or np.cos(eccAnomaly) != cosE:
            eccAnomaly = np.mod(eccAnomaly+np.pi,2*np.pi)
        
        date = np.mod(eccAnomaly - self.__eccentricity * sinE, 2*np.pi)/(2*np.pi)*(self.__period*self.daysInOneYear) + self.__periastronTime
        if output == 'jd':        
            return date
        elif output == 'iso':
            return timeConverter(time=date,output='iso')
            
           
# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------     
    def randomPositions(self,timeStart,timeEnd,number,layout='uniform',outputFormat='iso',pointOfView='earth'):
        """ Under construction 
        Summary::
        
        Arguments::
        
        Return::
        
        """
        timeStartJD = Time(timeStart,format='iso',scale='utc').jd
        timeEndJD = Time(timeEnd,format='iso',scale='utc').jd
        if timeStartJD > timeEndJD:
            print("timeStart should be < than timeEnd. The times have been automatically inverted")
            timeStart, timeEnd = timeEnd, timeStart
            timeStartJD, timeEndJD = timeEndJD, timeStartJD
        
        if layout == 'uniform':        
            timeJDSequence = np.linspace(timeStartJD,timeEndJD,number)
        elif layout == 'random':
            timeJDSequence = timeStartJD + np.random.rand(1,number)[0] * (timeEndJD - timeStartJD)

        if outputFormat == 'iso':
            timeJDSequence = Time(timeJDSequence,format='jd',scale='utc').iso.tolist()
        elif outputFormat == 'jd':
            timeJDSequence = timeJDSequence.tolist()
        
        k = 0
        trueAnomalySequence = np.zeros(len(timeJDSequence))
        for z in timeJDSequence:
            trueAnomalySequence[k] = Orbit.trueAnomaly(self,z)
            k += 1
            
        positionSequence = Orbit.positionOnOrbit(self,trueAnomalySequence,pointOfView,'arcsec') 
        result = {'timePosition' : timeJDSequence}  
        result.update(positionSequence)
        return result


# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------ 
    def chiSquare(self,obs,error={}): # This method has been added in the class StatisticMCMC and should be used from that way
        """ Under construction 
        Summary::
        
        Arguments::
        
        Return::
        
        """  
        if len(error) == 0:
            error = {'decError' : np.ones(obs['decPosition'].size), 'raError' : np.ones(obs['raPosition'].size)}
            
        timeSequence = obs['timePosition']
        obsPositionSequence = {'decPosition' : obs['decPosition'], 'raPosition' : obs['raPosition']}
        # TODO: old version. Now I should use list comprehension     
        k = 0
        trueAnomalySequence = np.zeros(len(timeSequence))        
        for i in timeSequence:
            trueAnomalySequence[k] = Orbit.trueAnomaly(self,i)
            k += 1
            
        modelPositionSequence = Orbit.positionOnOrbit(self,trueAnomalySequence,'earth','arcsec')
        
        chi2Dec =   np.sum(np.power(np.abs(obsPositionSequence['decPosition'] - modelPositionSequence['decPosition']) / error['decError'] , 2)) 
        chi2Ra =   np.sum(np.power(np.abs(obsPositionSequence['raPosition'] - modelPositionSequence['raPosition']) / error['raError'] , 2)) 

        
        return chi2Dec+chi2Ra
        

# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------
#    def transit(self, inclination, periastron, periastronTime):
#        if np.abs(inclination - 90) > 0.1:
#            return False
#        else:
#            taEarthTransit = np.mod(270 - periastron,360) # gives the true anomaly of the transit or occultation
#            taEarthOccu = np.mod(90 - periastron,360)
            

# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------
    def period(self, semiMajorAxis = 1, starMass = 1):
        """
        """
        periodSecSquared = (np.power(semiMajorAxis * const.au.value,3) / (starMass * const.M_sun.value)) * 4 * np.power(np.pi,2) / const.G.value
        period = np.power(periodSecSquared,1./2.) / (365.*24*3600)
        return period
        
# ------------------------------------------------------------------------------------------------------------------------------------------
    def semimajoraxis(self, period = 1, starMass = 1):
        """
        """
        aAUcubic = const.G.value * (starMass * const.M_sun.value) / (4 * np.power(np.pi,2)) * (period * 365.*24*3600)**2
        a = np.power(aAUcubic,1./3.) / const.au.value
        return a        
