# -*- coding: utf-8 -*-

#import math
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime


class FileHandler:
    """ DocString:  
        ----------
        Summary::   
        
        Note::
                                        
    """
    
    def __init__(self,filePath,addContent=[]): 
        """ The constructor of the class """
        # -----------------------
        # Preamble instructions
        # -----------------------
        #fileAlreadyExist = 0
        try:
            fileObject = open(filePath,'r')
            content = fileObject.readlines()
            #fileAlreadyExist = 1
        except IOError: # if the file doesn't exist, we create it (empty)
            answer = 'y'#str(input("The file {0} doesn't exist. Do you want to create a new empty txt file with the same name? ('y' / 'n') ".format(filePath)))
            if answer == 'y':
                fileObject = open(filePath,'w')
                content = list()
            elif answer == 'n':
                print("The file has not been created. The object cannot be created neither.")
                raise IOError("No such file has been found")
            else:
                print("You must choose between 'y' for yes and 'n' for no. The file has not been created. The object cannot be created neither.")
                raise IOError("No such file has been found")
        finally:
            fileObject.close()
            
        
        # -----------------------------
        # Main part of the constructor
        # -----------------------------
        self.__filePath = filePath
        self.__content = content
        self.__addContent = addContent
        #self.fileAlreadyExist = fileAlreadyExist        
        
    
    # -----------------------------          
    # Property 
    # -----------------------------   
    # TODO: property !!!!       
    def _get_content(self):
        with open(self.__filePath,'r') as fileObject:
            self.__content = fileObject.readlines()
        return self.__content      
        
# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------    
    def printContent(self): 
        """ 
        Summary:: 
            Print each line contained in the file
        Arguments::
            None
        Return::
            None         
        """    
        for i in self.__content:
            print(i)


# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------    
    def writeText(self, text): 
        """ 
        Summary:: 
            Print each line contained in the file
        Arguments::
            None
        Return::
            None         
        """    
        with open(self.__filePath,'a') as fileObject:
            if isinstance(text,str):
                fileObject.write("%s \n" % text)
            elif isinstance(text,tuple):
                defFormat = "%s"
                for k in range(1,len(text)):
                    defFormat += "\t %s"
                fileObject.write(defFormat % text)
                    
            
            
# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------   
    def writeContent(self,addNewContentList=[],addNewContentDict={}, addNewError={'decError': np.ones(0), 'raError': np.ones(0)}):   
        """ 
        Under construction
        Summary:: 
            Write the timePosition, \Delta RA (arcsec) + error, and \Delta DEC (arcsec) + error 
            in a file. These informations can be input either with a list or a dict (typically, 
            the return object from the method Orbit.randomPositions(...))
        Arguments::
            
        Return::
                     
        """ 
                
        
        with open(self.__filePath,'a') as fileObject:        
            if len(addNewContentList) != 0:
                for i in addNewContentList:
                    fileObject.write("%s" % i)
            
            if len(addNewContentDict) != 0:
                l = len(addNewContentDict['timePosition'])
                k = 0 
                
                if len(addNewError['decError']) == 1 and addNewError['decError'] == 0:
                    while k < l:                    
                        fileObject.write("%s\t\t%s\t%s\t%s\t%s\n" % (addNewContentDict['timePosition'][k],addNewContentDict['raPosition'][k],0.0,addNewContentDict['decPosition'][k],0.0))
                        k += 1            
                else:
                   while k < l:                    
                    fileObject.write("%s\t\t%s\t%s\t%s\t%s\n" % (addNewContentDict['timePosition'][k],addNewContentDict['raPosition'][k],addNewError['raError'][k],addNewContentDict['decPosition'][k],addNewError['decError'][k]))
                    k += 1                  


# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------
    def writeInfoSimu(self, info=[tuple()]):
        """ 
        Under construction
        Summary:: 

        Arguments::
            
        Return::
                     
        """ 
        header = [  '************************************',\
                    '**  MCMC simulation informations  **',\
                    '************************************'] 
        with open(self.__filePath,'a') as f:
            for i in range(len(header)):                
                f.write("%s\n" % header[i])
            
            now = datetime.datetime.now()
            now = '{}-{}-{} {}h{}m{}s'.format(now.year,now.month,now.day,now.hour,now.minute,now.second)
            f.write("{}\t\t{} \n \n".format('File creation:',now))
            
            for k in range(0,len(info)):
                f.write("%s\t%s\n" % info[k])
                
            listdir = list()
            listdir = os.listdir(info[-1][1])
            #print(listdir)
            f.write("\n-- %s \n" % 'List of files --')
            
            for j in range(0,len(listdir)):
                f.write("%s \n" % listdir[j])
# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------
    def getInfo(self): 
        """ 
        Under Construction
        Summary:: 
            Create a dict-type file from a txt file, which contains the timePosition, 
            \Delta RA (arcsec) + error, and \Delta DEC (arcsec) + error. The output layout
            is the same as the one given by the method Orbit.randomPositions(...).
        Arguments::
            
        Return::
                     
        """    
        info = FileHandler._get_content(self)
        l = len(info)
        ra, dec = (np.zeros(l),np.zeros(l))
        raError, decError = (np.zeros(l),np.zeros(l))
        keyTime = list()
        j = 0
        while j < l:
            k = 0
            dictTemp = dict()
            strTemp = str()
            for i in info[j]:
                if i == '\t' or i == '\n':
                    dictTemp[k] = strTemp
                    strTemp = str()
                    k += 1
                else:
                    strTemp += i
            
            ra[j] = float(dictTemp[2])
            dec[j] = float(dictTemp[4])
            keyTime.append(dictTemp[0])
            raError[j] = float(dictTemp[3])
            try:
                decError[j] = dictTemp[5]
            except KeyError:
                strTemp = str()                
                for i in info[j][::-1]:
                    if i == '\t' or i == '\n':
                        decError[j] = float(strTemp[::-1])
                        break
                    else:
                        strTemp += i        
            j += 1
        
        return {'timePosition' : keyTime, 'decPosition' : dec, 'raPosition' : ra},{'decError' : decError, 'raError' : raError}


# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------
    def showData(self, ob, er, lim=[(-1,1),(-1,1)]):
        """ 
        Under Construction
        Summary:: 
        
        Arguments::
            
        Return::
                     
        """ 
        plt.hold(True)
        
        
        #plt.plot(ob['raPosition'],ob['decPosition'],'ro')
        plt.errorbar(ob['raPosition'], ob['decPosition'], yerr=[er['decError'], er['decError']], xerr=[er['raError'], er['raError']], fmt='.')        
        plt.plot(0,0,'ko')
        
        plt.xlim([lim[0][0],lim[0][1]])
        plt.ylim([lim[1][0],lim[1][1]])
        
        plt.axes().set_aspect('equal')
        plt.gca().invert_xaxis()
        plt.title("Point of view: earth.\n North is top, East is left.")
        plt.xlabel("RA (arcsec)")
        plt.ylabel("DEC (arcsec)")
        plt.show()


# ------------------------------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------------------------------
    def extractFromTxt(self, keyword, filename, directoryRelativePath):
        """ 
        Summary:: 
            
        Arguments::
            
        Return::
                     
        """  
        print(self.__content)        

        

        

