'''
This is a set of data transform methods for the Kaggle competition
each of these methods accepts an epilepsy packet and returns a feature
vector or another epilepsy packet.

September 2014
'''

#Hack necessary for ipython on my mac - remove in over version
import os
if os.getlogin() == 'christopheguerrier':
  import sys 
  sys.path.append('/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')
  

import epilepsyTools as eT
import scipy.io as sci
from scipy import stats
import numpy as np
#Issue with sklearn import on my machine
from sklearn import preprocessing
#import pywt
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

'''FFT'''
def FFTBySquib(packet):
  if type(packet) is eT.EEGpackage:
    print "Warning, expect packet not package."
    return FFTBySquib(packet.packet)
  elif type(packet) is np.ndarray:
    features = []
    for squib in packet:
      features.append(np.fft.rfft(squib))
  else:
    print "Error invalid input."
    return False
  return features
#OK


'''Standardization
return a standardized EEGPacket'''
def Standardize(package):
  if type(package) is eT.EEGpackage:
    #Duplicate package
    newPac = package
    newPac.packet = preprocessing.scale(package.packet)
    return newPac
  else:
    print "Error wrong object input"
    return False
#OK

"""
Take a slice of the data on the last axis.
e.g. Slice(1, 48) works like a normal python slice, that is 1-47 will be taken
"""
def Slice(data, start, end):
  s = [slice(None),] * data.ndim
  s[-1] = slice(start, end)
  return data[s]
#OK

'''Low pass filter, return EEGPacket'''
def LowFilter(package,cutoff):
  if type(package) is eT.EEGpackage:
    #Duplicate package
    newPac = package
    data = newPac.packet
    nyq = cutoff / 2.0
    cutoffF = min(cutoff, nyq-1)
    h = signal.firwin(numtaps=101, cutoff=cutoffF, nyq=nyq)

    # data[i][ch][dim0]
    for i in range(len(data)):
        data_point = data[i]
        for j in range(len(data_point)):
            data_point[j] = signal.lfilter(h, 1.0, data_point[j]) #selected axis is out of range
    newPac.packet = data #bad data returned
    return newPac
  else:
    print "Error wrong object input"
    return False


"""
Mel-frequency cepstrum coefficients
"""
def MFCC(package):
  if type(package) is eT.EEGpackage:
    all_ceps = []
    for ch in package.packet:
      ceps, mspec, spec = mfcc(ch) #mfcc not defined
      all_ceps.append(ceps.ravel())
    return np.array(all_ceps)
  else:
    return False

"""
Take magnitudes of Complex data
"""
def Magnitude(package):
  if type(package) is eT.EEGpackage:
    data = package.packet
  else:
    data = package
  return np.absolute(data)
#OK


"""
Take the magnitudes and phases of complex data and append them together.
"""
def MagnitudeAndPhase(package):
  if type(package) is eT.EEGpackage:
    data = package.packet
  else:
    data = package
  magnitudes = np.absolute(data)
  phases = np.angle(data)
  return np.concatenate((magnitudes, phases), axis=1)
#OK

"""
    Apply Log10
"""
def Log10(package):
  if type(package) is eT.EEGpackage:
    data = package.packet
  else:
    data = package
  # 10.0 * log10(re * re + im * im)
  indices = np.where(data <= 0)
  data[indices] = np.max(data)
  data[indices] = (np.min(data) * 0.1)
  return np.log10(data)
#OK but reurn -Inf a lot !!


"""
    Subtract the mean, then take (min, max, standard_deviation) for each channel.
"""
def Stats(package):
  data = package.packet
  # data[ch][dim]
  shape = data.shape
  out = np.empty((shape[0], 3))
  for i in range(len(data)):
    ch_data = data[i]
    ch_data = data[i] - np.mean(ch_data)
    outi = out[i]
    outi[0] = np.std(ch_data)
    outi[1] = np.min(ch_data)
    outi[2] = np.max(ch_data)
  return out
#OK

"""
Resample time-series data.
"""
def Resample(package, sample_rate):
  data = package.packet
  axis = data.ndim - 1
  if data.shape[-1] > sample_rate:
    return resample(data, sample_rate, axis=axis)
  return data
#OK


"""
Resample time-series data using a Hanning window
"""
def ResamplingHanning(package, sample_rate):
  data = package.packet
  axis = data.ndim - 1
  out = resample(data, sample_rate, axis=axis, window=hann(M=data.shape[axis]))
  return out
#OK

"""
Daubechies wavelet coefficients. For each block of co-efficients
take (mean, std, min, max)
"""
def DaubWaveletStats(package,n):
  data = package.packet
  # data[ch][dim0]
  shape = data.shape
  out = np.empty((shape[0], 4 * (n * 2 + 1)), dtype=np.float64)
  def set_stats(outi, x, offset):
    outi[offset*4] = np.mean(x)
    outi[offset*4+1] = np.std(x)
    outi[offset*4+2] = np.min(x)
    outi[offset*4+3] = np.max(x)
    for i in range(len(data)):
      outi = out[i]
      new_data = pywt.wavedec(data[i], 'db%d' % n, level=n*2)
      for i, x in enumerate(new_data):
        set_stats(outi, x, i)
  return out
#OK
#DaubWaveletStats(mat,4)

"""
    Scale across the last axis.
"""
def UnitScale(package):
  data = package.packet
  return preprocessing.scale(data, axis=data.ndim-1)
#OK

"""
    Scale across the first axis, i.e. scale each feature.
"""
def UnitScaleFeat(package):
  data = package.packet
  return preprocessing.scale(data, axis=0)
#OK


"""
    Calculate correlation coefficients matrix across all EEG channels.
"""
def CorrelationMatrix(package):
  data = package.packet
  return np.corrcoef(data)
#OK

"""
    Take eigenvalues of a matrix, and sort them by magnitude in order to
    make them useful as features (as they have no inherent order).
"""
def Eigenvalues(matrix):
  w, v = np.linalg.eig(matrix)
  w = np.absolute(w)
  w.sort()
  return w
#ok

# Take the upper right triangle of a matrix
def upper_right_triangle(matrix):
    accum = []
    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[1]):
            accum.append(matrix[i, j])

    return np.array(accum)
#ok



"""
    Calculate overlapping FFT windows. The time window will be split up into num_parts,
    and parts_per_window determines how many parts form an FFT segment.

    e.g. num_parts=4 and parts_per_windows=2 indicates 3 segments
    parts = [0, 1, 2, 3]
    segment0 = parts[0:1]
    segment1 = parts[1:2]
    segment2 = parts[2:3]

    Then the features used are (segment2-segment1, segment1-segment0)

    NOTE: Experimental, not sure if this works properly.
"""
def OverlappingFFTDeltas(package, num_parts, start, end, parts_per_window):
  data = package.packet
  axis = data.ndim - 1
  parts = np.split(data, num_parts, axis=axis)

  #if slice end is 208, we want 208hz
  partial_size = (1.0 * parts_per_window) / num_parts
  #if slice end is 208, and partial_size is 0.5, then end should be 104
  partial_end = int(end * partial_size)
  partials = []
  for i in range(num_parts - parts_per_window + 1):
    combined_parts = parts[i:i+parts_per_window]
    if parts_per_window > 1:
      d = np.concatenate(combined_parts, axis=axis)
    else:
      d = combined_parts
    d = Slice(np.fft.rfft(d, axis=axis),start, partial_end)
    d = Magnitude(d)
    d = Log10(d)
    partials.append(d)
  diffs = []
  for i in range(1, len(partials)):
    diffs.append(partials[i] - partials[i-1])
  return np.concatenate(diffs, axis=axis)
#OK


def FFTWithOverlappingFFTDeltas(package, num_parts, parts_per_window, start, end):
  """
  As above but appends the whole FFT to the overlapping data.
  NOTE: Experimental, not sure if this works properly.
  """
  data = package
  axis = data.ndim - 1
  full_fft = np.fft.rfft(data, axis=axis)
  full_fft = Magnitude().apply(full_fft)
  full_fft = Log10().apply(full_fft)
  parts = np.split(data, num_parts, axis=axis)
  #if slice end is 208, we want 208hz
  partial_size = (1.0 * parts_per_window) / num_parts
  #if slice end is 208, and partial_size is 0.5, then end should be 104
  partial_end = int(end * partial_size)
  partials = []
  for i in range(num_parts - parts_per_window + 1):
    d = np.concatenate(parts[i:i+parts_per_window], axis=axis)
    d = Slice(np.fft.rfft(d, axis=axis),start, partial_end)
    d = Magnitude(d)
    d = Log10(d)
    #d = Slice(start, partial_end).apply(np.fft.rfft(d, axis=axis))
    #d = Magnitude().apply(d)
    #d = Log10().apply(d)
    partials.append(d)
  out = [full_fft]
  for i in range(1, len(partials)):
    out.append(partials[i] - partials[i-1])
  return np.concatenate(out, axis=axis)
#OK


  """
    Correlation in the frequency domain. First take FFT with (start, end) slice options,
    then calculate correlation co-efficients on the FFT output, followed by calculating
    eigenvalues on the correlation co-efficients matrix.

    The output features are (fft, upper_right_diagonal(correlation_coefficients), eigenvalues)

    Features can be selected/omitted using the constructor arguments.
    """
def FreqCorrelation(package, start, end, scale_option, with_fft=False, with_corr=True, with_eigen=True):
        #assert scale_option in ('us', 'usf', 'none')
        #assert with_corr or with_eigen
  data = package.packet
  data1 = FFT().apply(data)
  data1 = Slice(start, end).apply(data1)
  data1 = Magnitude().apply(data1)
  data1 = Log10().apply(data1)
  data2 = data1
  if scale_option == 'usf':
    data2 = UnitScaleFeat().apply(data2)
  elif scale_option == 'us':
    data2 = UnitScale().apply(data2)
  data2 = CorrelationMatrix().apply(data2)
  if with_eigen:
    w = Eigenvalues().apply(data2)
  out = []
  if with_corr:
    data2 = upper_right_triangle(data2)
    out.append(data2)
  if with_eigen:
    out.append(w)
  if with_fft:
    data1 = data1.ravel()
    out.append(data1)
  for d in out:
    assert d.ndim == 1
  return np.concatenate(out, axis=0)
#ISSUES


"""
Correlation in the time domain. First downsample the data, then calculate correlation co-efficients
followed by calculating eigenvalues on the correlation co-efficients matrix.

The output features are (upper_right_diagonal(correlation_coefficients), eigenvalues)

Features can be selected/omitted using the constructor arguments.
"""
def TimeCorrelation(package, max_hz, scale_option, with_corr=True, with_eigen=True):
    #assert scale_option in ('us', 'usf', 'none')
    #assert with_corr or with_eigen
    # so that correlation matrix calculation doesn't crash
    data = package.packet
    for ch in data:
        if np.alltrue(ch == 0.0):
            ch[-1] += 0.00001

    data1 = data
    if data1.shape[1] > max_hz:
        data1 = Resample(max_hz).apply(data1)

    if scale_option == 'usf':
        data1 = UnitScaleFeat().apply(data1)
    elif scale_option == 'us':
        data1 = UnitScale().apply(data1)

    data1 = CorrelationMatrix().apply(data1)

    if with_eigen:
        w = Eigenvalues().apply(data1)

    out = []
    if with_corr:
        data1 = upper_right_triangle(data1)
        out.append(data1)
    if with_eigen:
        out.append(w)

    for d in out:
        assert d.ndim == 1

    return np.concatenate(out, axis=0)
#ISSUES


"""
Combines time and frequency correlation, taking both correlation coefficients and eigenvalues.
"""
def TimeFreqCorrelation(package, start, end, max_hz, scale_option):
    data = package.packet
    data1 = TimeCorrelation(max_hz, scale_option).apply(data)
    data2 = FreqCorrelation(start, end, scale_option).apply(data)
    assert data1.ndim == data2.ndim
    return np.concatenate((data1, data2), axis=data1.ndim-1)
#ISSUES


"""
Combines FFT with time and frequency correlation, taking both correlation coefficients and eigenvalues.
"""
def FFTWithTimeFreqCorrelation(package, start, end, max_hz, scale_option):
    data = package.packet
    data1 = TimeCorrelation(max_hz, scale_option).apply(data)
    data2 = FreqCorrelation(start, end, scale_option, with_fft=True).apply(data)
    assert data1.ndim == data2.ndim
    return np.concatenate((data1, data2), axis=data1.ndim-1)
#ISSUES



def HillTFWTFC(package):
  return FFTWithTimeFreqCorrelation(package, 1, 48, 400, 'usf')
#ISSUES



'''Testing'''
#mat = eT.readMatReturnPackage("/Users/christopheguerrier/work/Kaggle/Epilepsy/Dog_5/Dog_5_interictal_segment_0389.mat")
#print FFTBySquib(Standardize(mat))

