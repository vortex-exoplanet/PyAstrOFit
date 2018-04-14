# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 08:31:34 2014

@author: Olivier
"""
from __future__ import print_function

# ------------------------------------------------------------------------------------------------------------------------------------------
#           IMPORT MODULES           
# ------------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import pickle
import OrbitAnalysis.StatisticsMCMC as sm
import OrbitAnalysis.FileHandler as fh
import matplotlib.pyplot as plt
from OrbitAnalysis.Toolbox import Toolbox as toolbox
import OrbitAnalysis.Orbit as o
from OrbitAnalysis.Diagnostics import Diagnostics as diag
#from astropy.time import Time
import os
#import itertools
from pyPdf import PdfFileWriter, PdfFileReader
#import itertools
import triangle
import math


# ------------------------------------------------------------------------------------------------------------------------------------------
#               MAIN         
# ------------------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------
#               USER PARAMETERS           
# --------------------------------------------
rootPath = 'runOutput/Gather_2015_1_11_11h24m52s/'
#'runOutput/Gather_2015_1_11_11h24m52s/'
#'runOutput/Gather_2015_1_23_8h57m5s/'
#Gather_2015_1_20_13h50m53s/'
#Gather_2015_1_22_13h47m29s/'
#Gather_2015_1_11_11h24m52s/   --> beta Pic b 5 (84 chains, #79000)
#Run_2015_1_12_19h44m44s/      --> beta Pic b 6 (25 chains, #61000)


directoryRelativePath = rootPath+'mcmcRun/'
info = 'runInfo.txt'
# --------------------------------------------
#              PARTICULAR OBJECTS            
# --------------------------------------------
stat = sm.StatisticsMCMC()
tool = toolbox()
diag = diag()

# --------------------------------------------
#    FILE TO ANALYZE - GET THE INFORMATIONS           
# --------------------------------------------
p = 6 # Number of model parameters (eccentricity, semiMajorAxis, ...)
fInfo = fh.FileHandler(os.path.join(directoryRelativePath,info))
content = fInfo._get_content()
for j in range(len(content)):
    print(content[j].split("\n")[0])        
print("\n")


filePath = tool.extractFromTxt('Data source file',info,directoryRelativePath)
mcNumber = int(tool.extractFromTxt('Markov chain number',info,directoryRelativePath))
prior = dict()
prior['starMass'] = float(tool.extractFromTxt('Star mass prior',info,directoryRelativePath))
prior['dStar'] = float(tool.extractFromTxt('Star distance prior',info,directoryRelativePath))
l = int(tool.extractFromTxt('Markov chain dim',info,directoryRelativePath))
scaleOptimal = tool.extractFromTxt('beta_mu fine tuning',info,directoryRelativePath)



key = str()
j = 0
while key != '-- List of files -- ':
    key = content[j].split("\n")[0]
    j += 1
varName = [content[i].split("\n")[0] for i in range(j,len(content))]
nFile = len(content)-1-j

# TODO: this part of the code (if) should be more general.
if scaleOptimal == 'Load': 
    tempPrerun = tool.findRunFromData(filePath, options={'AR': 'Only'})
    nFilePrerun = len(tempPrerun)
    prerunDirectory = tempPrerun[:21]


# -------------------------------
#      CREATE ANALYSE FOLDER           
# -------------------------------
directoryAnalyse = rootPath + 'mcmcAnalyze/'
subdirFigures = directoryAnalyse+'figures/'
subdirARAll = directoryAnalyse+'figures/AcceptanceRateMonitoring_All/'
subdirPDAll = directoryAnalyse+'figures/Posterior_Distribution_All/'
subdirCHAll = directoryAnalyse+'figures/Chi2_Map_All/'
subdirPDind = directoryAnalyse+'figures/Posterior_Distribution_perChains/'

tree = [directoryAnalyse,subdirARAll,subdirPDAll,subdirCHAll,subdirPDind]

for path in tree:
    if not os.path.exists(path):
        os.makedirs(path)


# -------------------------------
#           FUNCTIONS           
# -------------------------------
def append_pdf(input,output):
    [output.addPage(input.getPage(page_num)) for page_num in range(input.numPages)]



# ------------------------------------------------------------------------------------------------------------------------------------------
#           MAIN CODE
# ------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------
#           GET THE DATA           
# -------------------------------
print('Get the data ...')
f = fh.FileHandler(filePath) # creating an FileHandler object
(ob, er) = f.getInfo()
print('DONE ! \n')


raMin = np.amin(ob['raPosition'])
raMax = np.amax(ob['raPosition'])
decMin = np.amin(ob['decPosition'])
decMax = np.amax(ob['decPosition'])

xlim1 = raMin-0.1*(raMax-raMin)
xlim2 = raMax+0.1*(raMax-raMin)
ylim1 = decMin-0.1*(decMax-decMin)
ylim2 = decMax+0.1*(decMax-decMin)

f.showData(ob,er, lim = [(xlim1,xlim2),(ylim1,ylim2)])

dof = float(len(ob['timePosition'])*2 - p)

# -------------------------------------------
#       WHICH TEST DO YOU WANT TO RUN ?
# -------------------------------------------
acceptancerateTest = False # TODO: save the arReject object in a file. It is required for the gelmanrubinTest.


gelmanrubinTest = False # TODO: try to load the arReject object usin pickler.
#lagautocorrelation = False


inferenceTest = True
highProbableSolution = True
inferenceTest_corr = False
twoDimPDF = False
individualPosterior = False


displayTest = False
histoTest = False

chi2Histo = False

initialstateTest = False

transitTest = False

displayAll = False

# -------------------------------
#       Extract the output 
# -------------------------------
# Run outputs
if l == 0: # If the runInfo.txt returns l = 0, it means that markovChainRun = False ==> no Markov chain run. 
    raise Exception("The dimension of the Markov chains is {}, which implies that no run has been started. There is nothing to test...".format(l))


# Common files to load
fileToLoad_common = set(['run_pKey'])

# Inititialization
fileToLoad_acceptanceRate = set()
fileToLoad_gelmanrubinTest = set()
fileToLoad_inferenceTest = set()
fileToLoad_chi2Test = set()
fileToLoad_transitTest = set()
creatingComplete = False
fileToLoadAdd = list()

if acceptancerateTest:
    fileToLoad_acceptanceRate = set(['run_pKey','run_acceptanceRateAll','run_chiSquareAll'])

if gelmanrubinTest:
    fileToLoad_gelmanrubinTest = set(['run_markovChainAll'])    
    
if inferenceTest:
    fileToLoad_inferenceTest = set(['run_pKey','run_markovChainAllComplete','run_markovChainTrueComplete','run_chiSquareAllComplete','run_markovChainAll','run_arReject'])    

if chi2Histo:
    fileToLoad_chi2Test = set(['run_pKey','run_chiSquareAllComplete'])
    
if transitTest:
    fileToLoad_transitTest = set(['run_pKey','run_markovChainAllComplete','run_markovChainTrueComplete'])    

fileToLoad = list(fileToLoad_common | fileToLoad_acceptanceRate | fileToLoad_gelmanrubinTest | fileToLoad_inferenceTest | fileToLoad_chi2Test | fileToLoad_transitTest)


objName = dict()
for key in fileToLoad:
    try:
        with open(os.path.join(directoryRelativePath,key),'rb') as fileRead:
            myPickler = pickle.Unpickler(fileRead)
            objName[key[4:]] = myPickler.load()  
        print("The file -{}- has been found.".format(key))    
    except:
        print("The file -{}- doesn't exist in the folder. It will be created.".format(key))
        creatingComplete = True
        try:
            with open(os.path.join(directoryRelativePath,'run_arReject'),'rb') as fileRead:
                myPickler = pickle.Unpickler(fileRead)
                arReject = myPickler.load() 
        except:
            if not acceptancerateTest:
                raise Exception("The file -run_arReject-, required to construct the markovChain(All/True)Complete, hasn't been found in the folder. You should run an -acceptanceRateTest- test before trying a -inferenceTest- test.")

        

pKey = objName['pKey']

try:
    markovChainAll = objName['markovChainAll']
except:
    if creatingComplete:
        with open(os.path.join(directoryRelativePath,'run_markovChainAll'),'rb') as fileRead:
            myPickler = pickle.Unpickler(fileRead)
            markovChainAll = myPickler.load()        

try:
    acceptanceRateAll = objName['acceptanceRateAll']
except:
    pass

try:
    chiSquareAll = objName['chiSquareAll']
except:
    if creatingComplete:
        with open(os.path.join(directoryRelativePath,'run_chiSquareAll'),'rb') as fileRead:
            myPickler = pickle.Unpickler(fileRead)
            chiSquareAll = myPickler.load()

try:
    whichParamAll = objName['whichParamAll']
except:
    if creatingComplete:
        with open(os.path.join(directoryRelativePath,'run_whichParamAll'),'rb') as fileRead:
            myPickler = pickle.Unpickler(fileRead)
            whichParamAll = myPickler.load()    

try:
    markovChainAllComplete = objName['markovChainAllComplete']
    markovChainTrueComplete = objName['markovChainTrueComplete']
    chiSquareAllComplete = objName['chiSquareAllComplete']
except:
    pass


try:
    arReject = objName['arReject']
except:
    pass

try:
    chiSquareAllComplete = objName['chiSquareAllComplete']
except:
    pass


# ------------------------------------------------------------------------------------------------------------------------------------------
#          
# ------------------------------------------------------------------------------------------------------------------------------------------
"""
acceptancerateTest
    - For a given run, one determines the acceptance rate for all the parameters and for all the Markov chains. 
    - The figures are saved in a directory.
Under construction
"""

# ------------------------------------------------------------------------------------------------------------------------------------------
#          
# ------------------------------------------------------------------------------------------------------------------------------------------
print("\n---------- RESULTS -----------\n")

######################################################################
###   ACCEPTANCE RATE PICTURES + INFOS (chi2, mean, median, ...)   ###
######################################################################
if acceptancerateTest:
    print("\n")
    arLength = len(acceptanceRateAll['0'][pKey[0]])
    chiSquareMinAll = dict()
    arMedianAll = dict()
    arMeanAll = dict()
    arMeanPerMC = dict()
    arStdAll = dict()
    arReject = list()
    ar_output = PdfFileWriter()
    for k in range(0,mcNumber):
        print('--------------------------------')
        print("Markov Chain {}\n".format(k))
        chiSquareMinAll[str(k)] = np.amin(chiSquareAll[str(k)][1:]) / dof
        arMedianAll[str(k)] = np.median(acceptanceRateAll[str(k)].values())    
        arMeanAll[str(k)] = np.mean(acceptanceRateAll[str(k)].values())
        arMeanPerMC[str(k)] = [np.mean([acceptanceRateAll[str(k)][kkey][q] for kkey in pKey]) for q in range(arLength)] 
        arStdAll[str(k)] = np.std(acceptanceRateAll[str(k)].values())
        print('minimum chi2 is equal to {}'.format(chiSquareMinAll[str(k)]))
        print('mediane AR: {}, mean AR: {}, std AR: {}'.format(arMedianAll[str(k)],arMeanAll[str(k)],arStdAll[str(k)]))
        print('median AR per parameter: a={:.3f} | e={:.3f} | i={:.3f} | omega={:.3f} | w={:.3f} | tp={:.3f}'.format(np.median(acceptanceRateAll[str(k)][pKey[0]]),\
                                                                                                                        np.median(acceptanceRateAll[str(k)][pKey[1]]),\
                                                                                                                        np.median(acceptanceRateAll[str(k)][pKey[2]]),\
                                                                                                                        np.median(acceptanceRateAll[str(k)][pKey[3]]),\
                                                                                                                        np.median(acceptanceRateAll[str(k)][pKey[4]]),\
                                                                                                                        np.median(acceptanceRateAll[str(k)][pKey[5]])))
        # Acceptance rate monitoring                                                                                                                 
        fig = plt.figure()
        plt.plot(acceptanceRateAll[str(k)]['semiMajorAxis'],'b',acceptanceRateAll[str(k)]['eccentricity'],'r',acceptanceRateAll[str(k)]['inclinaison'],'g',\
        acceptanceRateAll[str(k)]['longitudeAscendingNode'],'y',acceptanceRateAll[str(k)]['periastron'],'m', acceptanceRateAll[str(k)]['periastronTime'],'c')
        
        plt.plot([0,arLength],[0.2,0.2],'--k')
        plt.plot([0,arLength],[0.4,0.4],'--k')
        
        plt.xlim([0,arLength])        
        plt.ylim([0,1])
        
        plt.xlabel('step number')
        plt.ylabel('Acceptance rate')        
        plt.suptitle("ACCEPTANCE RATE MONITORING\n", fontsize=14, fontweight='bold')
        plt.figtext(.14,.85,"Markov chain #{}".format(k))   
        
        plt.legend(['$a$','$e$','$i$','$\Omega$','$\omega$','$t_\omega$'], fontsize='xx-small')
        
        plt.savefig(subdirARAll+'arMonitoring_MC{}.pdf'.format(k))
        if displayAll:
            plt.show()
        else:
            plt.close(fig)
        
        append_pdf(PdfFileReader(file(subdirARAll+'arMonitoring_MC{}.pdf'.format(k))),ar_output)
        print('--------------------------------')

    for key in arMedianAll.keys():
        if arMedianAll[key] < 0.2 or arMedianAll[key] > 0.5 or arStdAll[key] > 0.1 or chiSquareMinAll[key] > 2.:
            print("The run #{} should be rejected".format(key))
            arReject.append(int(key))
    
    with open(os.path.join(directoryRelativePath,'run_arReject'),'wb') as fileSave:
        myPickler = pickle.Pickler(fileSave)
        myPickler.dump(arReject)
        
    # Mean acceptance rate between all parameters, for each Markov chain
    arMeanPerMCAll = [np.mean([arMeanPerMC[str(k)][i] for k in range(mcNumber)]) for i in range(arLength)]        
    fig = plt.figure()
    plt.hold('on')
    plt.plot(arMeanPerMCAll,'r')
    for k in range(mcNumber):
        plt.plot(arMeanPerMC[str(k)],'b') 
    plt.plot(arMeanPerMCAll,'r')
    
    plt.plot([0,arLength],[0.2,0.2],'--k')
    plt.plot([0,arLength],[0.4,0.4],'--k')
    plt.xlim([0,arLength])        
    plt.ylim([0,1])  
    
    plt.xlabel('step number')
    plt.ylabel('Mean acceptance rate')        
    plt.suptitle("MEAN ACCEPTANCE RATE MONITORING\n", fontsize=12, fontweight='bold')
    plt.figtext(.14,.85,"All Markov chains")
    
    plt.legend(['Mean of the blue curves','Mean between AR for all parameters'], fontsize='xx-small')    
    
    plt.savefig(subdirFigures+'arMonitoring_Mean.pdf')
    if displayAll:
        plt.show()
    else:
        plt.close(fig)
        
    
    ar_output.write(file(subdirFigures+'arMonitoring_All.pdf',"w"))    
######################################################################
######################################################################


# ----------------------------------
#   COMBINE THE MARKOV CHAIN         
# ----------------------------------

#markovChainAllComplete = {key: np.concatenate(tuple(markovChainAll[str(nu)][key] for nu in range(mcNumber))) for key in pKey}
#markovChainAll = markovChainAllComplete

# extract the updated (or not) elements from the Markov Chain, i.e. we reject the value of a given step if it hadn't been considered (due to the Gibbs Sampler):
if creatingComplete:
    markovChainTrue = {str(nu) : dict() for nu in range(mcNumber)}
    del(i,j)
    validMarkovChain = list(set(range(mcNumber)) - set(arReject))
    for nu in validMarkovChain:     
        for key in pKey:
            markovChainTrue[str(nu)][key] = markovChainAll[str(nu)][key][[i for i, j in enumerate(whichParamAll[str(nu)]) if j == key]]
    
    markovChainAllComplete = {key: np.concatenate(tuple(markovChainAll[str(nu)][key] for nu in validMarkovChain)) for key in pKey}
    markovChainTrueComplete = {key: np.concatenate(tuple(markovChainTrue[str(nu)][key] for nu in validMarkovChain)) for key in pKey} 
    chiSquareAllComplete = [np.concatenate(tuple(chiSquareAll[str(nu)] for nu in validMarkovChain))][0]
    
    
    with open(os.path.join(directoryRelativePath,'run_markovChainAllComplete'),'wb') as fileSave:
        myPickler = pickle.Pickler(fileSave)
        myPickler.dump(markovChainAllComplete)
        
    with open(os.path.join(directoryRelativePath,'run_markovChainTrueComplete'),'wb') as fileSave:
        myPickler = pickle.Pickler(fileSave)
        myPickler.dump(markovChainTrueComplete)    
    
    with open(os.path.join(directoryRelativePath,'run_chiSquareAllComplete'),'wb') as fileSave:
        myPickler = pickle.Pickler(fileSave)
        myPickler.dump(chiSquareAllComplete)  



            
            
##################################
###   Gelman-Rubin statistic   ### 
##################################
if gelmanrubinTest:
    
    print("\n")
    print("*****************************")
    print("***   GELAMN-RUBIN TEST   ***")
    print("*****************************")
    print("\n")
    
    try:
        diag.gelman_rubin(pKey,markovChainAll,reject=arReject)
    except NameError:
        diag.gelman_rubin(pKey,markovChainAll,reject=[])
    


#    combs = list()      
#    combs = [[list(x) for x in itertools.combinations(range(mcNumber), i)] for i in range(2,mcNumber+1)]
#
#    
#    #whichChain = range(mcNumber)
#    whichChain = [1,2]
#    
#    zAll = {str(k): dict() for k in range(mcNumber)}
#    uKey = ['u0','u1','u2','u3','u4','u5']
#    for k in range(0,mcNumber):  
#        for j in range(p):
#            zAll[str(k)][uKey[j]] = markovChainAll[str(k)][pKey[j]]
#    
#    zStatistic = dict()
#    lc = float(len(zAll['0']['u1']))
#    for j in uKey:
#        zStatistic[j] = {'zcMean' : np.empty(0), 'zcVar' : np.empty(0), 'zW' : np.empty(0), 'zMeanTotal' : np.empty(0), 'zB' : np.empty(0), 'zVar' : np.empty(0), 'rGelmanRubin' : np.empty(0)}
#        for k in whichChain:
#            zStatistic[j]['zcMean'] = np.append(zStatistic[j]['zcMean'], np.mean(zAll[str(k)][j]))
#            zStatistic[j]['zcVar'] = np.append(zStatistic[j]['zcVar'], np.var(zAll[str(k)][j], ddof=1))
#        
#        zStatistic[j]['zW'] = np.mean(zStatistic[j]['zcVar'])
#        zStatistic[j]['zMeanTotal'] = np.mean(zStatistic[j]['zcMean'])
#        zStatistic[j]['zB'] = np.var(zStatistic[j]['zcMean'],ddof=1) * lc
#        zStatistic[j]['zVar'] = (lc - 1)/lc * zStatistic[j]['zW'] + zStatistic[j]['zB'] / lc
#        zStatistic[j]['rGelmanRubin'] = np.sqrt(zStatistic[j]['zVar'] / zStatistic[j]['zW'])
#    
#    #for i in range(0,mcNumber):
#    #    tool.showHistogram(vector = zAll[str(i)]['u1'])
#    
#    #tool.showHistogram(vector = [zAll[str(u)]['u1'] for u in range(0,mcNumber)])
#            
#    rGelRub = np.zeros(p)
#    j = 0
#    for i in uKey:
#        rGelRub[j] = zStatistic[i]['rGelmanRubin']
#        j += 1
#        
#    print("\n Gelman-Rubin values for the 6 parameters : {}".format(rGelRub))    
###################################
###################################

########################################
###     Lag autocorrelation test     ###
########################################
#if lagautocorrelation:
#    n = 40
#    lag = np.array(np.linspace(0,0.75,n))
#    plt.figure()
#    f ,pl = plt.subplots(3,2)
#    f.tight_layout()
#    for j in range(p):
#        lline = j // 2
#        cline = j % 2
#        w = diag.lagAutocorrelation(markovChainAll['0'][pKey[j]],n)
#        pl[lline,cline].plot(lag,w)
#        pl[lline,cline].set_title(pKey[j])
#
#

###################################
###################################


  
  
 
###########################   
####   INFERENCE TEST   ###
########################### 

                   
if inferenceTest:      
   
   ### Determine the set of parameters for which the chi2 is minimum

    # Min chi2 orbit
    paramTitle = {pKey[j]: ['$a$','$e$','$i$','$\Omega$','$\omega$','$t_\omega$'][j] for j in range(p)}
    if 'chiSquareAllComplete' not in locals():
        chiSquareAllComplete = [np.concatenate(tuple(chiSquareAll[str(nu)] for nu in range(mcNumber)))][0]
    temp = np.where(chiSquareAllComplete == 0)
    for x in temp:
        chiSquareAllComplete[x] = 10000 * (1 + np.random.rand(1))
    indexMinChi2 = chiSquareAllComplete.argmin()
    parametersMinChi2 = {key: markovChainAllComplete[key][indexMinChi2] for key in pKey}
    
    orbitMinChi2 = o.Orbit(semiMajorAxis = parametersMinChi2['semiMajorAxis'],\
                            eccentricity = parametersMinChi2['eccentricity'],\
                            inclinaison = parametersMinChi2['inclinaison'],\
                            longitudeAscendingNode = parametersMinChi2['longitudeAscendingNode'],\
                            periastron = parametersMinChi2['periastron'],\
                            periastronTime = parametersMinChi2['periastronTime'],\
                            dStar = prior['dStar'],starMass = prior['starMass']) 
#    orbitMinChi2.showOrbit('earth','arcsec',\
#                            addPosition=ob['timePosition'],\
#                            addPoints=[ob['raPosition'],ob['decPosition'],er['raError'],er['decError']],\
#                            lim = [(-0.3,-0.10),(-0.4,-0.25)],  link = True, title="Beta Pic b   ($\chi^2_r$ min)", save=(True,subdirFigures+'orbitChi2Min_crop1'))
                            
    orbitMinChi2.showOrbit('earth','arcsec',\
                            addPosition=ob['timePosition'],\
                            addPoints=[ob['raPosition'],ob['decPosition'],er['raError'],er['decError']],\
                            lim = [(xlim1,xlim2),(ylim1,ylim2)],  link = True, title="Beta Pic b   ($\chi^2_r$ min)", save=(True,subdirFigures+'orbitChi2Min_crop2'))                            
                            
    orbitMinChi2.showOrbit('earth','arcsec',\
                            addPosition=ob['timePosition'],\
                            addPoints=[ob['raPosition'],ob['decPosition'],er['raError'],er['decError']],\
                            link = True, title="Beta Pic b   ($\chi^2_r$ min)", save=(True,subdirFigures+'orbitChi2Min_full'))                             


   
   
   
   ### Determine the most probable solution for all the parameters
    xHistF = dict()
    binsF = dict()
    nHistF = dict()
    pourcentBothPart = dict()
    arg_max = dict()
    height_max = dict()
    val_max = dict()
    confidenceInterval = dict()
    confidenceInterval2 = dict()
    confidence = 68.
    for key in pKey:            
        xHistF[key], binsF[key] = np.histogram(markovChainTrueComplete[key],bins = 1000,density=False)
        nHistF[key] = float(np.sum(xHistF[key]))
        arg_max[key] = xHistF[key].argmax()
        height_max[key] = xHistF[key][arg_max[key]]
        val_max[key] = binsF[key][arg_max[key]]
        
        pourcentBothPart[key] = np.array([np.sum(xHistF[key][:arg_max[key]])/nHistF[key],np.sum(xHistF[key][arg_max[key]:])/nHistF[key]])
        
        # method #1
        pourcentage = 0.
        k = 0
        while pourcentage < confidence:
            arg_bound = np.where(xHistF[key]>height_max[key]-k)[0]
            pourcentage = np.sum(xHistF[key][arg_bound])/nHistF[key]*100.
            k += 1
            
        confidenceInterval[key] = binsF[key][arg_bound[[0,-1]]]


        # method #2
#        confidenceInterval2[key] = np.empty(2)
#        pourcentage = 0.
#        k = 0
#        while pourcentage < (confidence/100.*pourcentBothPart[key][0])*100.:
#            temp_midPart = xHistF[key][0:arg_max[key]]
#            l_midPart = len(temp_midPart)
#            pourcentage = np.sum(temp_midPart[l_midPart-k:-1])/nHistF[key]*100.
#            k += 1                
#        confidenceInterval2[key][0] = binsF[key][arg_max[key]-k] 
#        
#        pourcentage = 0.
#        k = 0
#        while pourcentage < (confidence/100.*pourcentBothPart[key][1])*100.:
#            temp_midPart = xHistF[key][arg_max[key]+1:-1]
#            l_midPart = len(temp_midPart)
#            pourcentage = np.sum(temp_midPart[:k])/nHistF[key]*100.
#            k += 1                
#        confidenceInterval2[key][1] = binsF[key][arg_max[key]+k]
       

    # Highly probable orbit 
#    orbitHighly = o.Orbit(semiMajorAxis = parametersMinChi2['semiMajorAxis'],\
#                            eccentricity = val_max['eccentricity'],\
#                            inclinaison = val_max['inclinaison'],\
#                            longitudeAscendingNode = val_max['longitudeAscendingNode'],\
#                            periastron = val_max['periastron'],\
#                            periastronTime = parametersMinChi2['periastronTime'],\
#                            dStar = prior['dStar'],starMass = prior['starMass']) 
#    orbitHighly.showOrbit('earth','arcsec',\
#                            addPosition=ob['timePosition'],\
#                            addPoints=[ob['raPosition'],ob['decPosition'],er['raError'],er['decError']],\
#                            lim = [(-0.3,-0.10),(-0.4,-0.25)],  link = True, title="Beta Pic b   (Highly probable)", save=(True,subdirFigures+'orbithighly')) 
#
#    orbitHighly.chiSquare(ob,er) / dof
   
 
    ### Highly probable solutions
    #confidenceInterval['periastron'][1] = 10
    #confidenceInterval['semiMajorAxis'][0] = 8.8   
    if highProbableSolution:
        nOrbit = 30000
        parameter = {key: np.diff(confidenceInterval[key])*np.random.rand(nOrbit)+confidenceInterval[key][0] for key in pKey}
        #chi2Orbit = np.empty(0)
        chi2Orbit = 1000.
        temp = 0.
        for j in range(nOrbit):
            orbit = o.Orbit(semiMajorAxis = parameter['semiMajorAxis'][j],\
                        eccentricity = parameter['eccentricity'][j],\
                        inclinaison = parametersMinChi2['inclinaison'],\
                        longitudeAscendingNode = parameter['longitudeAscendingNode'][j],\
                        periastron = parameter['periastron'][j],\
                        periastronTime = parameter['periastronTime'][j],\
                        dStar = prior['dStar'],starMass = prior['starMass'])
            #chi2Orbit = np.append(chi2Orbit,orbit.chiSquare(ob,er)) 
            temp = orbit.chiSquare(ob,er)/dof
            if chi2Orbit > temp:
                chi2Orbit = temp
                parameterMin = {key: parameter[key][j] for key in pKey}
                print('New chi2r: {}'.format(chi2Orbit))
    
                
                
        orbit = o.Orbit(semiMajorAxis = parameterMin['semiMajorAxis'],\
                    eccentricity = parameterMin['eccentricity'],\
                    inclinaison = parameterMin['inclinaison'],\
                    longitudeAscendingNode = parameterMin['longitudeAscendingNode'],\
                    periastron = parameterMin['periastron'],\
                    periastronTime = parameterMin['periastronTime'],\
                    dStar = prior['dStar'],starMass = prior['starMass'])  
    
        if chi2Orbit < 1:
            orbit.showOrbit('earth','arcsec', addPosition=ob['timePosition'],addPoints=[ob['raPosition'],ob['decPosition'],er['raError'],er['decError']], lim = [(-0.3,-0.10),(-0.4,-0.25)], link = True, title="Beta Pic b", save=(True,subdirFigures+'orbitHighProb.pdf')) 
                
            for key in pKey:
                if key != 'periastronTime':
                    print('{} = {}'.format(key,parameterMin[key]))
                else:
                    print('{} = {}'.format(key,tool.timeConverter(parameterMin[key],output='iso')))  
       
        confidenceInterval['periastron'][1] = 94
        confidenceInterval['semiMajorAxis'][0] = 8.89-0.42
        
        
   
   ### Construct the figures for the posterior distribution    
    pdf_output = PdfFileWriter()    
    
    xHist = dict()
    bins = dict()
    for key in pKey:
        fig = plt.figure()     
        plt.hold('on')
        xHist[key], bins[key] = np.histogram(markovChainTrueComplete[key],bins = 150,density=False)
        
        if key == pKey[4] and np.amax(markovChainTrueComplete['periastron']) > 180:        
            plt.plot((bins[key][:-1] + bins[key][1:])/2-180,xHist[key],'b')
            plt.xlim([-180,180])
            plt.plot((val_max[key]-180)*np.ones(2),[0,np.amax(xHist[key])],'g')
        else:
            plt.plot((bins[key][:-1] + bins[key][1:])/2,xHist[key],'b')
            plt.plot(val_max[key]*np.ones(2),[0,np.amax(xHist[key])],'g')
            #if key == pKey[0]:
               # plt.xlim([7,15])

        plt.plot(parametersMinChi2[key]*np.ones(2),[0,np.amax(xHist[key])],'--r')
        try:
            plt.plot(parameterMin[key]*np.ones(2),[0,np.amax(xHist[key])],'--y')
        except:
            pass
      
        try:            
            if key != pKey[4] and key != pKey[5]:                
                plt.plot(confidenceInterval[key][0]*np.ones(2),[0,np.amax(xHist[key])],'g', linestyle='dotted')
                plt.plot(confidenceInterval[key][1]*np.ones(2),[0,np.amax(xHist[key])],'g', linestyle='dotted') 
                plt.plot(confidenceInterval2[key][0]*np.ones(2),[0,np.amax(xHist[key])],'m', linestyle='dotted')
                plt.plot(confidenceInterval2[key][1]*np.ones(2),[0,np.amax(xHist[key])],'m', linestyle='dotted')                                
        except:
            pass
    
        plt.ylim([0,np.amax(xHist[key])])
        
        plt.xlabel(paramTitle[key])
        plt.ylabel('$N_{Orbits}$', fontsize=10)
        plt.suptitle('POSTERIOR DISTRIBUTION  - '+paramTitle[key]+' -', fontsize=12, fontweight='bold')
        plt.savefig(subdirPDAll+'posteriorDistribution_'+key+'.pdf')
        
        if displayAll:
            plt.show()
        else:
            plt.close(fig)
            
        append_pdf(PdfFileReader(file(subdirPDAll+'posteriorDistribution_'+key+'.pdf')),pdf_output)   
    
    pdf_output.write(file(subdirFigures+'posteriorDistribution_All.pdf',"w"))
            
    print('the ch2rDistribution figure has been saved')

    with open(os.path.join(directoryRelativePath,'run_histogram'),'wb') as fileSave:
    	myPickler = pickle.Pickler(fileSave)
	myPickler.dump(xHist)
    
    with open(os.path.join(directoryRelativePath,'run_bins'),'wb') as fileSave:
	myPickler = pickle.Pickler(fileSave)
	myPickler.dump(bins)






   ### Construct the figures for the posterior distribution 
    if individualPosterior:
        validMarkovChain = list(set(range(mcNumber)) - set(arReject))
        
        for nu in validMarkovChain:    
            #nu = 0
               
            pdf_output = PdfFileWriter()    
            
            xHist = dict()
            bins = dict()
            for key in pKey:
                fig = plt.figure()     
                plt.hold('on')
                xHist[key], bins[key] = np.histogram(markovChainAll[str(nu)][key],bins = 100,density=False)
                
                if key == pKey[4] and np.amax(markovChainAll[str(nu)]['periastron']) > 180:        
                    plt.plot((bins[key][:-1] + bins[key][1:])/2-180,xHist[key],'b')
                    plt.xlim([-180,180])
                    plt.plot((val_max[key]-180)*np.ones(2),[0,np.amax(xHist[key])],'g')
                else:
                    plt.plot((bins[key][:-1] + bins[key][1:])/2,xHist[key],'b')
                    plt.plot(val_max[key]*np.ones(2),[0,np.amax(xHist[key])],'g')
                    #if key == pKey[0]:
                       # plt.xlim([7,15])
            
                plt.ylim([0,np.amax(xHist[key])])
                
                plt.xlabel(paramTitle[key])
                plt.ylabel('$N_{Orbits}$', fontsize=10)
                plt.suptitle('POSTERIOR DISTRIBUTION  - '+paramTitle[key]+' -', fontsize=12, fontweight='bold')
                plt.savefig(subdirPDind+'posteriorDistribution_'+str(nu)+'_'+key+'.pdf')
                
                if displayAll:
                    plt.show()
                else:
                    plt.close(fig)
                    
                append_pdf(PdfFileReader(file(subdirPDind+'posteriorDistribution_'+str(nu)+'_'+key+'.pdf')),pdf_output)   
            
            pdf_output.write(file(subdirPDind+'posteriorDistribution_'+str(nu)+'.pdf',"w"))
            
            for key in pKey:    
                os.remove(subdirPDind+'posteriorDistribution_'+str(nu)+'_'+key+'.pdf')
                    
            print('the posterior Distribution figure for the Markov chain #{} has been saved'.format(nu))










   ### Correlation figures
    if twoDimPDF:
        nCorr = 30000
        nP = 5
        index_corr = np.random.choice(np.arange(0,l,1),nCorr)
        sam = np.array([[markovChainAllComplete[key][j] for key in pKey[0:nP]] for j in index_corr])
        fig = triangle.corner(sam, labels=['$a$','$e$','$i$','$\Omega$','$\omega$','$t_\omega$'], truths=[parametersMinChi2[kk] for kk in pKey[0:nP]])
        fig.suptitle('MARGINALIZED PROBABILITY DISTRIBUTIONS', fontsize=14, fontweight='bold')
        plt.savefig(subdirFigures+'triangle.pdf')
    
    
    
    
    


    
    
    
    
    
    ### Write inference results in a text file
    fRes = fh.FileHandler(os.path.join(rootPath+'mcmcAnalyze','mcmcResults.txt'))
    fRes.writeText('###########################')
    fRes.writeText('####   INFERENCE TEST   ###')
    fRes.writeText('###########################')
    fRes.writeText(' ')
    fRes.writeText('Results of the MCMC fit')
    fRes.writeText('----------------------- ')
    fRes.writeText(' ')
    fRes.writeText('>> Orbit parameter values (highly probable):')
    fRes.writeText(' ')
    for i in range(p):
        confidenceMax = confidenceInterval[pKey[i]][1]-val_max[pKey[i]]
        confidenceMin = val_max[pKey[i]]-confidenceInterval[pKey[i]][0]
        if pKey[i] == 'longitudeAscendingNode':
            text = '{}: \t{:.2f} \t\t-{:.2f} \t\t+{:.2f}'
        elif pKey[i] == 'periastronTime':            
            text = '{}: \t\t{:.2f} \t-{:.2f} \t+{:.2f}'
        elif pKey[i] == 'eccentricity':
            text = '{}: \t\t\t{:.4f} \t\t-{:.4f} \t+{:.4f}'
        else:
            text = '{}: \t\t\t{:.2f} \t\t-{:.2f} \t\t+{:.2f}'
            
        fRes.writeText(text.format(pKey[i],val_max[pKey[i]],confidenceMin,confidenceMax))

    fRes.writeText(' ')
    fRes.writeText('>> Orbit parameter values (chi2r min):')
    fRes.writeText(' ')
    for i in range(p):
        if pKey[i] == 'longitudeAscendingNode':
            text = '{}: \t{:.2f} '
        elif pKey[i] == 'periastronTime':            
            text = '{}: \t\t{:.2f} '
        elif pKey[i] == 'eccentricity':
            text = '{}: \t\t\t{:.4f}'
        else:
            text = '{}: \t\t\t{:.2f} '
            
        fRes.writeText(text.format(pKey[i],parametersMinChi2[pKey[i]]))        
        
    
###########################
###########################


###########################   
####   TRANSIT TEST   ###
###########################
if transitTest:
    transit = np.empty(0)
    period = np.empty(0)
    nTest = 1000
    
    indexValidI = np.where(np.abs(markovChainTrueComplete['inclinaison']-90) < 180)[0]
    nIndex = len(indexValidI)
    
    
    indexTest = np.random.choice(np.arange(0,nIndex,1),nTest)
    for j in range(nTest):
        
        orbit = o.Orbit(semiMajorAxis = markovChainTrueComplete['semiMajorAxis'][indexValidI[indexTest[j]]],\
                            eccentricity = markovChainTrueComplete['eccentricity'][indexValidI[indexTest[j]]],\
                            inclinaison = markovChainTrueComplete['inclinaison'][indexValidI[indexTest[j]]],\
                            longitudeAscendingNode = markovChainTrueComplete['longitudeAscendingNode'][indexValidI[indexTest[j]]],\
                            periastron = markovChainTrueComplete['periastron'][indexValidI[indexTest[j]]],\
                            periastronTime = markovChainTrueComplete['periastronTime'][indexValidI[indexTest[j]]],\
                            dStar = prior['dStar'],starMass = prior['starMass'])                        
                            
        transit = np.append(transit,orbit.timeFromTrueAnomaly(math.radians(90-markovChainTrueComplete['periastron'][indexTest[j]]),output='jd'))
        period = np.append(period,orbit._get_period())
        #print(tool.timeConverter(orbit._get_periastronTime(),output='iso'),orbit._get_period())
    
    
    datemin = tool.timeConverter(np.amin(transit),output='iso')
    axemin = tool.timeConverter(time=datemin[0:4]+'-01-01 00:00:00.0',output='jd')
    
    transit2 = (transit-axemin)/365+float(datemin[0:4])
    #tool.showHistogram(vector=transit2,nBin = 200, title = 'Transit', save =(True,subdirFigures+'transit.pdf'))
    
    
    temp = np.where(transit2 < 2012)[0]
    transit2[temp] += period[temp]
    #tool.showHistogram(vector=transit2,nBin = 200, title = 'Transit', save =(True,subdirFigures+'transit.pdf'))
    
    transit2 = transit2[np.where(transit2 < 2022)[0]]
    period = period[np.where(transit2 < 2022)[0]]
    
    alias = transit2 - period
    alias2 = transit2 - 2*period
    tool.showHistogram(vector=np.concatenate((alias,alias2,transit2)),nBin = 200, title = 'Transit', xylabels = ('Transit date (yr-JD)','$N_{Orbits}$'), save =(True,subdirFigures+'transit.pdf'))

    with open(os.path.join(directoryRelativePath,'run_transit'),'wb') as fileSave:
	myPickler = pickle.Pickler(fileSave)
	myPickler.dump(np.concatenate((alias,alias2,transit2)))

###########################
###########################
    

########################################################
###   DISPLAY THE CURRENT HIGHLY PROBABLE SOLUTION   ###
########################################################
#if displayTest:
#    prior = {'starMass' : 1.75, 'dStar' : 19.3}
#    runName = 'test'
#    l = 7200
#    # finding the Markov chain with the smallest chi2
#    #indexMin = chiSquareMinAll.keys()[np.fromiter(chiSquareMinAll.values(), np.float).argmin()]
#    indexMin = '0'
#    
#    ### display the highly probable orbital solution
#    minP = dict()
#    for key in pKey:
#        minP[key] =  np.median(markovChainAll[indexMin][key])
#    
#    orbitMedian = o.Orbit(semiMajorAxis = minP['semiMajorAxis'], eccentricity = minP['eccentricity'], inclinaison = minP['inclinaison'],\
#    longitudeAscendingNode = minP['longitudeAscendingNode'], periastron = minP['periastron'], periastronTime = minP['periastronTime'], \
#    starMass = prior['starMass'], dStar = prior['dStar'])
#    
#    orbitMedian.showOrbit(pointOfView = 'earth', unit = 'arcsec', addPosition = ob['timePosition'], addPoints = [ob['raPosition'],ob['decPosition']],save=(False,filePath[:len(filePath)-4]+'_'+runName+'_'+indexMin+'_HP_earth'))  
#    orbitMedian.showOrbit(pointOfView = 'face-on', unit = 'au', addPosition = ob['timePosition'],save=(False,filePath[:len(filePath)-4]+'_'+runName+'_'+indexMin+'_HP_faceon')) 
#    # display the smallest chi2 solution
#    indexChi2Min = np.argmin(chiSquareAll[indexMin][1:l]) + 1
#    minPchi2 = {'semiMajorAxis' : markovChainAll[indexMin]['semiMajorAxis'][indexChi2Min], 'eccentricity' : markovChainAll[indexMin]['eccentricity'][indexChi2Min], 'inclinaison' : markovChainAll[indexMin]['inclinaison'][indexChi2Min],\
#    'longitudeAscendingNode' : markovChainAll[indexMin]['longitudeAscendingNode'][indexChi2Min], 'periastron' : markovChainAll[indexMin]['periastron'][indexChi2Min], 'periastronTime' : markovChainAll[indexMin]['periastronTime'][indexChi2Min], \
#    'starMass' : prior['starMass'], 'dStar' : prior['dStar']}         
#    orbitMin = o.Orbit(semiMajorAxis = markovChainAll[indexMin]['semiMajorAxis'][indexChi2Min], eccentricity = markovChainAll[indexMin]['eccentricity'][indexChi2Min], inclinaison = markovChainAll[indexMin]['inclinaison'][indexChi2Min],\
#    longitudeAscendingNode = markovChainAll[indexMin]['longitudeAscendingNode'][indexChi2Min], periastron = markovChainAll[indexMin]['periastron'][indexChi2Min], periastronTime = markovChainAll[indexMin]['periastronTime'][indexChi2Min], \
#    starMass = prior['starMass'], dStar = prior['dStar']) 
#    
#    orbitMin.showOrbit(pointOfView = 'earth', unit = 'arcsec', addPosition = ob['timePosition'], addPoints = [ob['raPosition'],ob['decPosition']], save=(False,filePath[:len(filePath)-4]+'_'+runName+'_'+indexMin+'_chi2min_earth')) 
#    orbitMin.showOrbit(pointOfView = 'face-on', unit = 'au', addPosition = ob['timePosition'], save=(False,filePath[:len(filePath)-4]+'_'+runName+'_'+indexMin+'_chi2min_faceon')) 
######################################################
######################################################   
    
    
    
#chauvin = {'semiMajorAxis' : 8.8, 'eccentricity' : 0.021, 'inclinaison' : 88.5, 'longitudeAscendingNode' : -148.24+180, 'periastron' : -115+180, 'periastronTime' : 2453845.5}
#chauvinHist = {'semiMajorAxis' : (0,25), 'eccentricity' : (0,1), 'inclinaison' : (75,105), 'longitudeAscendingNode' : (15,55), 'periastron' : (0,180), 'periastronTime' : (2453845.5-1500,2453845.5+1500)}
##########################################
###   DISPLAY PARAMETRIZED HISTOGRAM   ###
########################################## 
#if histoTest:
#    index = '3'
#    param = 'longitudeAscendingNode'
#    units = '(deg)' 
#    save = False
#
#    tool.showHistogram(vector = markovChainAll[index][param], bins = 100)
#    
#    bin1 = chauvinHist[param][0]
#    bin2 = chauvinHist[param][1]
#    binNum = 100
#    xHist, bins = np.histogram(markovChainAll[index][param],bins = np.linspace(bin1,bin2,binNum), density=False)
#    plt.hold(True)
#    plt.plot(bins[0:binNum-1],xHist)
#    plt.plot([chauvin[param],chauvin[param]],[0,np.amax(xHist)*1.2],'--r')
#    plt.xlabel(param+'  '+units)
#    plt.ylabel('N orbits')
#    plt.ylim([0,np.amax(xHist)*1.2])
#    
#    if save:
#        plt.savefig(runName+'_'+'mc'+index+'_'+param+'.pdf')
#    plt.show()
    

#################################
###   DISPLAY ch2 HISTOGRAM   ###
#################################    

if chi2Histo:
    print('#################################')
    print('###   DISPLAY ch2 HISTOGRAM   ###')
    print('#################################')
    #if 'chiSquareAllComplete' not in locals():
    #    chiSquareAllComplete = [np.concatenate(tuple(chiSquareAll[str(nu)] for nu in range(mcNumber)))][0]

    #save = False
    
    #tool.showHistogram(vector = chiSquareAllComplete/dof, nBin = np.linspace(0,10,100))
    displayAll = True
    bin1, bin2, binNum = (0,10,500)
    xHist, bins = np.histogram(chiSquareAllComplete[2000000:]/dof,bins = np.linspace(bin1,bin2,binNum), density=False)
    
    with open(os.path.join(directoryRelativePath,'run_chi2histogram'),'wb') as fileSave:
	myPickler = pickle.Pickler(fileSave)
	myPickler.dump(xHist)

    with open(os.path.join(directoryRelativePath,'run_chi2bins'),'wb') as fileSave:
	myPickler = pickle.Pickler(fileSave)
	myPickler.dump(bins)


    
    fig = plt.figure()    
    #plt.axes([0,0,1,0.85])    
    plt.hold(True)
    plt.plot(bins[0:binNum-1],xHist)
    plt.xscale('log', nonposy='clip')
    plt.xlim([0.1,10])
    plt.xlabel('$\chi^2_r$', fontsize=12)
    plt.ylabel('N counts')
    plt.suptitle("$\chi^2_r$ DISTRIBUTION\n", fontsize=14, fontweight='bold')

    plt.savefig(subdirFigures+'chi2rDistribution.pdf')
    print('the ch2rDistribution figure has been saved')
    if displayAll:
        plt.show()
    else:
        plt.close(fig)
    
    
####################################
###   INITIALSTATE DISTRIBUtION  ###
####################################
if initialstateTest:
    
    ### Scale outputs
    if scaleOptimal == 'True' or scaleOptimal == 'Only': 
        with open(os.path.join(rootPath+'scaleParameterPhase','scale_candidatAll'),'rb') as fileRead:
            myPickler = pickle.Unpickler(fileRead)
            scale_candidatAll = myPickler.load()
            
        with open(os.path.join(rootPath+'scaleParameterPhase','scale_scaleParameterAll'),'rb') as fileRead:
            myPickler = pickle.Unpickler(fileRead)
            scale_scaleParameterAll = myPickler.load()    
        
        runSC = True
    elif scaleOptimal == 'Load':
        scale_initialState = {pKey[j]: np.empty(0) for j in range(p)}
        
        for prekey in prerunDirectory:
            prePath = 'runOutput/'+prekey+'mcmcRun'
            with open(os.path.join(prePath,'run_initialState'),'rb') as fileRead:
                myPickler = pickle.Unpickler(fileRead)
                temp_scale_initialState = myPickler.load() 
            for key in pKey:
                scale_initialState[key] = np.concatenate((scale_initialState[key],temp_scale_initialState[key]))           
    else:
        runSC = False        
    ###     
    
    par = (2,5)    
    lim = [(np.amin(scale_initialState[pKey[0]]),np.amax(scale_initialState[pKey[0]])),(0,1),(0,90),(0,90),(0,90),(np.amin(scale_initialState[pKey[5]]),np.amax(scale_initialState[pKey[5]]))]
    
    x = scale_initialState[pKey[par[0]]]
    y = scale_initialState[pKey[par[1]]]
    
    fig = plt.figure()
    plt.plot(x,y,'go')  
    
    plt.xlim([lim[par[0]][0],lim[par[0]][1]])    
    plt.ylim([lim[par[1]][0],lim[par[1]][1]])  
    
    plt.xlabel(pKey[par[0]])
    plt.ylabel(pKey[par[1]])
    plt.suptitle("INITIAL VALUES FOR THE MARKOV CHAINS\n", fontsize=12, fontweight='bold')
    
    
   

####################################
####################################    
    
    
    
    
    
