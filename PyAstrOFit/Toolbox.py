# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:52:30 2015

@author: Olivier
"""

from __future__ import division
from __future__ import print_function

__all__ = ['open_pickle',
           'save_pickle',
           'send_email',
           'cpu_count',
           'now',
           'radialToEq',
           'timeConverter',
           'playSound']

###############################################################################
###############################################################################
###############################################################################

# -----------------------------------------------------------------------------
def open_pickle(objectToOpen):
    """
    Open and return the content of a pickable object
    """
    import pickle
    with open(objectToOpen,'rb') as fileRead:
        myPickler = pickle.Unpickler(fileRead)
        res = myPickler.load()
    
    print('The file {} has been succesfully loaded.'.format(objectToOpen))
    return res
    


# -----------------------------------------------------------------------------
def save_pickle(objectToSave, path = None):
    """
    Pickle an object into a file.
    """
    import pickle
    
    if path is None:
        import datetime
        today = datetime.datetime.now()
        month = today.month
        day = today.day
        hour = today.hour
        minute = today.minute
        if len(str(month)) == 1:
            month = '0'+str(month)
            
        if len(str(day)) == 1:
            day = '0'+str(day)
            
        if len(str(hour)) == 1:
            hour = '0'+str(hour) 
            
        if len(str(minute)) == 1:
            minute = '0'+str(minute)            
            
        path = 'fileSave_{}{}{}_{}{}'.format(today.year,month,day,hour,minute)
        
    with open(path,'wb') as fileSave:
        myPickler = pickle.Pickler(fileSave)
        myPickler.dump(objectToSave)
        
    print('The file {} has been succesfully saved.'.format(path))


# -----------------------------------------------------------------------------
def send_email(sender,receiver,msg = None,subject = None):
    """
    """
    import smtplib
    from email.mime.text import MIMEText
    
    if msg is not None:
        if msg[len(msg)-3:] == 'txt':
            fp = open(msg,'rb')
            msg_to_send = MIMEText(fp.read())
            fp.close()
        else:
            msg_to_send = MIMEText(msg)
    else:
        msg_to_send = MIMEText('Hello Olivier,\n\nThe run is finished.\n\nBye,\nOlivier')
    
    msg_to_send['Subject'] = subject
    msg_to_send['From'] = sender
    msg_to_send['To'] = receiver
    
    s = smtplib.SMTP('smtp.ulg.ac.be')
    s.sendmail(sender,receiver,msg_to_send.as_string())
    s.quit()
    
    print('')
    print('The email has been successfully sent.')
    
    
# -----------------------------------------------------------------------------
def cpu_count():
    """ 
    Number of available virtual or physical CPUs on this system, i.e.
    user/real as output by time(1) when called with an optimally scaling
    userspace-only program
    """
    import os
    import re
    import subprocess
    
    # cpuset
    # cpuset may restrict the number of *available* processors
    try:
        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                      open('/proc/self/status').read())
        if m:
            res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
            if res > 0:
                return res
    except IOError:
        pass

    # Python 2.6+
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        pass

    # http://code.google.com/p/psutil/
    try:
        import psutil
        return psutil.cpu_count()   # psutil.NUM_CPUS on old versions
    except (ImportError, AttributeError):
        pass

    # POSIX
    try:
        res = int(os.sysconf('SC_NPROCESSORS_ONLN'))

        if res > 0:
            return res
    except (AttributeError, ValueError):
        pass

    # Windows
    try:
        res = int(os.environ['NUMBER_OF_PROCESSORS'])

        if res > 0:
            return res
    except (KeyError, ValueError):
        pass

    # jython
    try:
        from java.lang import Runtime
        runtime = Runtime.getRuntime()
        res = runtime.availableProcessors()
        if res > 0:
            return res
    except ImportError:
        pass

    # BSD
    try:
        sysctl = subprocess.Popen(['sysctl', '-n', 'hw.ncpu'],
                                  stdout=subprocess.PIPE)
        scStdout = sysctl.communicate()[0]
        res = int(scStdout)

        if res > 0:
            return res
    except (OSError, ValueError):
        pass

    # Linux
    try:
        res = open('/proc/cpuinfo').read().count('processor\t:')

        if res > 0:
            return res
    except IOError:
        pass

    # Solaris
    try:
        pseudoDevices = os.listdir('/devices/pseudo/')
        res = 0
        for pd in pseudoDevices:
            if re.match(r'^cpuid@[0-9]+$', pd):
                res += 1

        if res > 0:
            return res
    except OSError:
        pass

    # Other UNIXes (heuristic)
    try:
        try:
            dmesg = open('/var/run/dmesg.boot').read()
        except IOError:
            dmesgProcess = subprocess.Popen(['dmesg'], stdout=subprocess.PIPE)
            dmesg = dmesgProcess.communicate()[0]

        res = 0
        while '\ncpu' + str(res) + ':' in dmesg:
            res += 1

        if res > 0:
            return res
    except OSError:
        pass

    raise Exception('Can not determine number of CPUs on this system')    

# -----------------------------------------------------------------------------
def now():
    """
    """
    import datetime
    
    clock = datetime.datetime.now()  
    h = clock.hour if clock.hour >= 10 else '0'+str(clock.hour)
    m = clock.minute if clock.minute >= 10 else '0'+str(clock.minute) 
    s = clock.second if clock.second >= 10 else '0'+str(clock.second) 
    
    return [h,m,s]
    
    
# ------------------------------------------------------------------------------------------------------------------------------------------ 
def radialToEq(r = 1,t = 0, rError = 0, tError = 0, display = False):
    """ 
    Summary:: 
        Convert the position given in (r,t) into \delta RA and \delta DEC, as well as the corresponding uncertainties. 
        t = 0° (resp. 90°) points toward North (resp. East).            
    Arguments::
        
    Return::
                 
    """  
    import numpy as np
    import math
    import matplotlib.pyplot as plt
       
    ra = (r * np.sin(math.radians(t)))
    dec = (r * np.cos(math.radians(t)))   
    u, v = (ra, dec)
    
    nu = np.mod(np.pi/2.-math.radians(t), 2*np.pi)
    a, b = (rError,r*np.sin(math.radians(tError)))
    
    beta = np.linspace(0,2*np.pi,5000)
    x, y = (u + (a * np.cos(beta) * np.cos(nu) - b * np.sin(beta) * np.sin(nu)), v + (b * np.sin(beta) * np.cos(nu) + a * np.cos(beta) * np.sin(nu)))
    
    raErrorInf = u - np.amin(x)
    raErrorSup = np.amax(x) - u
    decErrorInf = v - np.amin(y)
    decErrorSup = np.amax(y) - v        

    if display:        
        plt.hold(True)
        plt.plot(u,v,'ks',x,y,'r')        
        plt.plot((r+rError) * np.cos(nu), (r+rError) * np.sin(nu),'ob',(r-rError) * np.cos(nu), (r-rError) * np.sin(nu),'ob')
        plt.plot(r * np.cos(nu+math.radians(tError)), r*np.sin(nu+math.radians(tError)),'ok')
        plt.plot(r * np.cos(nu-math.radians(tError)), r*np.sin(nu-math.radians(tError)),'ok')
        plt.plot(0,0,'og',np.cos(np.linspace(0,2*np.pi,10000)) * r, np.sin(np.linspace(0,2*np.pi,10000)) * r,'y')
        plt.plot([0,r*np.cos(nu+math.radians(tError*0))],[0,r*np.sin(nu+math.radians(tError*0))],'k')
        plt.axes().set_aspect('equal')
        lim = np.amax([a,b]) * 2.
        plt.xlim([ra-lim,ra+lim])
        plt.ylim([dec-lim,dec+lim])
        plt.gca().invert_xaxis()
        plt.show()
        
    return ((ra,np.mean([raErrorInf,raErrorSup])),(dec,np.mean([decErrorInf,decErrorSup])))  
     
# ------------------------------------------------------------------------------------------------------------------------------------------
def  timeConverter(time='2015-01-01 00:00:00.0', finput= '', output='jd'):
    """ 
    Summary:: 
        
    Arguments::
        
    Return::
                 
    """ 
    import numpy as np
    from astropy.time import Time
    
    def isLeapYear(x):
        if x%4 == 0 and x%100 != 0:
            return True
        elif x%400 == 0:
            return True
        else:
            return False
    
    # JD 1 jan 0001: 1721425.5
#        try:
    if finput == 'epoch' and not isinstance(time,str) and time < 2100 and output == 'iso':
        year = int(np.floor(time))                
        if not isLeapYear(year):                
            daysFloat =(time-year)*365
            days = int(np.floor(daysFloat))
            h = int(np.floor(24*(daysFloat-days)))
            monthsCumul = [0,31,59,90,120,151,181,212,243,273,304,334,365]
            if days == 0:
                days = 1
            m = np.sum(days > np.array(monthsCumul))
            d = days - monthsCumul[m-1]+1
            return '{}-{}-{} {}:00:00.0'.format(str(year), str(m) if m > 9 else '0'+str(m), str(d) if d > 9 else '0'+str(d), str(h) if h > 9 else '0'+str(h))   
        else:
            daysFloat =(time-year)*366                
            days = int(np.floor(daysFloat))
            h = int(np.floor(24*(daysFloat-days)))
            monthsCumul = [0,31,60,91,121,152,182,213,244,274,305,335,366]
            if days == 0:
                days = 1
            m = np.sum(days > np.array(monthsCumul))
            d = days - monthsCumul[m-1]+1
            return '{}-{}-{} {}:00:00.0'.format(str(year), str(m) if m > 9 else '0'+str(m), str(d) if d > 9 else '0'+str(d), str(h) if h > 9 else '0'+str(h)) 
            
            
    if output == 'jd' and isinstance(time,str):
        return Time(time,format='iso',scale='utc').jd
    elif output == 'jd' and not isinstance(time,str):
        return time
    elif output == 'iso' and not isinstance(time,str):
        return Time(time,format='jd',scale='utc').iso
    elif output == 'iso' and isinstance(time,str):
        return time
    elif output == 'epoch' and isinstance(time,str):
        y = int(time[0:4])
        m = int(time[5:7])
        d = int(time[8:10])
        if not isLeapYear(y):
            monthsCumul = [0,31,59,90,120,151,181,212,243,273,304,334,365]
            return y+(monthsCumul[m-1]+d)/365.
        elif isLeapYear(y):
            monthsCumul = [0,31,60,91,121,152,182,213,244,274,305,335,366]
            return y+(monthsCumul[m-1]+d)/366.                

# ------------------------------------------------------------------------------------------------------------------------------------------
def playSound(self, filePath=str()):
    """ 
    Summary:: 
        
    Arguments::
        
    Return::
                 
    """ 
    try:
        import wave
        import pyaudio
    except ImportError:
        print('The package wave and/or pyaudio must be installed.')
        return None
            
         
    chunk = 2048
    wf = wave.open(filePath, 'rb')
    p = pyaudio.PyAudio()
    
    stream = p.open(
        format = p.get_format_from_width(wf.getsampwidth()),
        channels = wf.getnchannels(),
        rate = wf.getframerate(),
        output = True)
    data = wf.readframes(chunk)
    
    while data != '':
        stream.write(data)
        data = wf.readframes(chunk)
    
    stream.close()
    p.terminate()                  