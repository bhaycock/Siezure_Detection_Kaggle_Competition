#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
"""
Barry Haycock
2014 / 09 / 21

Collection of methods that directly affect data as part of a multi-step transform.
"""
import scipy.io as sci
from scipy import stats
import numpy as np
#Issue with sklearn import on my machine
from sklearn import preprocessing
from scipy import signal
from scipy.signal import resample, hann

"""
Resample time-series data.
"""
def Resample(data, sample_rate):
  axis = data.ndim - 1
  if data.shape[-1] > sample_rate:
    return resample(data, sample_rate, axis=axis)
  return data

"""
FFT
"""
def FFT(data):
  axis = data.ndim - 1
  return np.fft.rfft(data, axis=axis)

"""
Resample time-series data.
"""
def Resample(data, sample_rate):
  axis = data.ndim - 1
  if data.shape[-1] > sample_rate:
    return resample(data, sample_rate, axis=axis)
  return data
"""
Take a slice of the data on the last axis.
e.g. Slice(1, 48) works like a normal python slice, that is 1-47 will be taken
"""
def Slice(data, start, end):
  s = [slice(None),] * data.ndim
  s[-1] = slice(start, end)
  return data[s]

"""
For testing... efficiently checks an array for infs and NaNs
"""
def checkNaNorInf(data):
  return (np.isnan(np.sum(data)) or np.isinf(np.sum(data)))

"""
Take magnitudes of Complex data
"""
def Magnitude(data):
  return np.absolute(data)

"""
    Apply Log10
"""
def Log10(data):
  # 10.0 * log10(re * re + im * im)
  indices = np.where(data <= 0)
  data[indices] = np.max(data)
  data[indices] = (np.min(data) * 0.1)
  return np.log10(data)

"""
    Scale across the last axis.
"""
def UnitScale(data):
  #data = package.packet  #Not sure about this, but I'm trying. Did same to UnitScaleFeat, below.
  return preprocessing.scale(data, axis=data.ndim-1)

"""
    Scale across the first axis, i.e. scale each feature.
"""
def UnitScaleFeat(data):
  return preprocessing.scale(data, axis=0)

"""
    Calculate correlation coefficients matrix across all EEG channels.
"""
def CorrelationMatrix(data):
#  data = package.packet    these mini-functions need to operate on specific matrices
  return np.corrcoef(data)

"""
    Take eigenvalues of a matrix, and sort them by magnitude in order to
    make them useful as features (as they have no inherent order).
"""
def Eigenvalues(matrix):
  w, v = np.linalg.eig(matrix)
  w = np.absolute(w)
  w.sort()
  return w

# Take the upper right triangle of a matrix
def upper_right_triangle(matrix):
  accum = []
  for i in range(matrix.shape[0]):
    for j in range(i+1, matrix.shape[1]):
      accum.append(matrix[i, j])

  return np.array(accum)

