'''
This is a set of data transform methods for the Kaggle competition
each of these methods accepts an epilepsy packet and returns a feature
vector or another epilepsy packet.

September 2014.

How transforms are tested:
#Mininum test
eT.testTransform(transforName)
'''

#Hack necessary for ipython on my mac - remove in over version
import warnings
import os
if os.getlogin() == 'christopheguerrier':
  import sys 
  sys.path.append('/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')
  
#Import
import epilepsyTools as eT
import scipy.io as sci
from scipy import signal
from scipy import stats
import numpy as np
from sklearn import preprocessing
from scipy import signal
from scipy.signal import resample, hann
# optional modules for trying out different transforms
try:
    import pywt
except ImportError, e:
    pass

try:
    from scikits.talkbox.features import mfcc
except ImportError, e:
    pass


'''List transforms
- FFTBySquib (initially from Hills )
- dwtBySquib : discrete waveleet transform, return cA coefficient
- SquibDWTFFT : per squib dwt + fft

'''


'''FFT'''
#def FFTBySquib(packet):
#  if type(packet) is eT.EEGpackage:
#    #print "Warning, expect packet not package."
#    return FFTBySquib(packet.packet)
#  elif type(packet) is np.ndarray:
#    features = []
#    for squib in packet:
#      features.extend(np.fft.rfft(squib))
#  else:
#    print "Error invalid input."
#    return False
#  return features

#Unit test
#eT.testTransform(FFTBySquib, tryPlot = False)
#OK

#def dwtBySquib(packet):
#  if type(packet) is eT.EEGpackage:
#    #print "Warning, expect packet not package."
#    return dwtBySquib(packet.packet)
#  elif type(packet) is np.ndarray:
#    features = []
#    for squib in packet:
#      cA, cD = pywt.dwt(squib, 'db2')
#      features.extend(cA)
#      features.extend(cD)
#  else:
#    print "Error invalid input."
#    return False
#  return features

#Unit test
#eT.testTransform(dwtBySquib, tryPlot = False)
#OK

''' This function merge the squib of a packet point by point and return 
a numpy array, a FFT transform and discret wavelet transform. '''

#def avgSquibFFT(packet):
#  #average squib point by point - can be splitted by side
#  if type(packet) is eT.EEGpackage:
#    #print "Warning, expect packet not package."
#    return avgSquibFFT(packet.packet)
#  elif type(packet) is np.ndarray:
#    features = []
#    nchan, npts = packet.shape
#    average = packet.mean(axis=0)
#    features.append(np.fft.rfft(average,axis=0))
#    return features
#  else:
#    print "Error invalid input."
#    return False


#Unit test
#check = eT.testTransform(avgSquibFFT, tryPlot = False)
#OK
#Read Directory And Return Transformed List Took 1.13507914543 seconds 
#         This corresponds to a time of  0.189179857572 per read-and-transform operation.
def getWavelet(data, w, level = 14):
  mode = pywt.MODES.sp1
  #w = 'coif5' #"DWT: Signal irregularity"
  #w = 'sym5'  #"DWT: Frequency and phase change - Symmlets5")
  w = pywt.Wavelet(w)
  a = data
  allc = []
  for i in xrange(level):
    (a, d) = pywt.dwt(a, w, mode)
    #print len(a)
    #print len(d)
  allc.extend(a.tolist())
  allc.extend(d.tolist())
  return allc

def getFFT(data, window_size):
  win=signal.hann(window_size)    
  yf = np.fft.fft(data)
  freqs = np.fft.fftfreq(len(data))
  #for coef,freq in zip(yf,freqs):
  #  if coef:
  #      print('{c:>6} * exp(2 pi i t * {f})'.format(c=coef,f=freq))
  #print(freqs.min(),freqs.max())
  # (-0.5, 0.499975)
  # Find the peak in the coefficients
  frate=11025.0
  yf=yf[1:]#be sure to remove 0hz
  idx=np.argmax(np.abs(yf)**2)
  freq=freqs[idx]
  freq_in_hertz=abs(freq*frate)
  #print(freq_in_hertz)
  return freq_in_hertz 
  

def SquibDWTFFT(packet):
  if type(packet) is eT.EEGpackage:
    return SquibDWTFFT(packet.packet)
  elif type(packet) is np.ndarray:
    features = []
    #number of sample in the n second window
    nchan, npts = packet.shape
    for squib in packet:
      #features.extend(getWavelet(squib, 'coif5', 18))
      features.extend(getWavelet(squib, 'sym5', 18))
      features.append(getFFT(squib, 399))
      features.append(squib.max())
      features.append(squib.mean())
      features.append(squib.std())
      features.append(squib.min())
      #print len(features)
    #print "XXXXXXXX"
    #print len(features)
  return features
      

#Unit test
#check = eT.testTransform(SquibDWTFFT, tryPlot = False)

def avgSquibDWT(packet):
  #average squib point by point - can be splitted by side
  if type(packet) is eT.EEGpackage:
    #print "Warning, expect packet not package."
    return avgSquibDWT(packet.packet)
  elif type(packet) is np.ndarray:
    features = []
    nchan, npts = packet.shape
    average = packet.mean(axis=0)
    #cA, cD = pywt.dwt(average,'db2')
    features.extend(getWavelet(average, 'coif5', 14))
    features.extend(getWavelet(average, 'sym5', 14))
    #if len(features) != 104:
    #  warnings.warn('Unexpected length for the feature')
    #check for Nan
    #na_list = np.isnan(features)
    #inf_list = np.isfinite(features)
    #replace Nan and infinite with 0
    np.set_printoptions(precision=8)
    feat = np.nan_to_num(features)
    return feat.tolist()
  else:
    print "Error invalid input."
    return False

#Unit test
#check = eT.testTransform(avgSquibDWT, tryPlot = False)
#OK
#Read Directory And Return Transformed List Took 0.503623008728 seconds 
#         This corresponds to a time of  0.0839371681213 per read-and-transform operation.

