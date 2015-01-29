#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
"""
Barry Haycock
2014 / 09 / 21

Hill's Time and Freq transformation, I don't know how well it will work, though.
"""
import scipy.io as sci
from scipy import stats
import numpy as np
#Issue with sklearn import on my machine
from sklearn import preprocessing
from scipy import signal
from scipy.signal import resample, hann

from dataManipulationMethods import *
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
        data1 = Resample(data1, max_hz)

    if scale_option == 'usf':
        data1 = UnitScaleFeat(data1)
    elif scale_option == 'us':
        data1 = UnitScale(data1)


    data1 = np.corrcoef(data1)

    if with_eigen:
        w = Eigenvalues(data1)
    out = []
    if with_corr:
        data1 = upper_right_triangle(data1)
        out.append(data1)
    if with_eigen:
        out.append(w)

    for d in out:
        assert d.ndim == 1

    return np.concatenate(out, axis=0)


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
  data1 = FFT(data)
  data1 = Slice(data1, start, end)
  data1 = Magnitude(data1)
  data1 = Log10(data1)
  data2 = data1
  if scale_option == 'usf':
    data2 = UnitScaleFeat(data2)
  elif scale_option == 'us':
    data2 = UnitScale(data2)
  data2 = CorrelationMatrix(data2)
  if with_eigen:
    w = Eigenvalues(data2)
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



def FFTWithTimeFreqCorrelation(package, start, end, max_hz, scale_option):  # Hill's winning function
  data1 = TimeCorrelation(package, max_hz, scale_option)
  data2 = FreqCorrelation(package, start, end, scale_option, with_fft=True)
  assert data1.ndim == data2.ndim
  return np.concatenate((data1, data2), axis=data1.ndim-1).tolist()




