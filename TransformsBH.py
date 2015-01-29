#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

""" This is a set of data transform methods for the Kaggle competition
each of these methods accepts an epilepsy packet and returns a feature
vector. In this way, the feature vectors can be added together as one 
see's fit by either calling each transform one at a time or by creating
a new transform that sums these guys together.

These are no out of date. See documentation. They're sceduled for deletion 
after I have finished this test.

Barry Haycock
September 2014.
"""

# All transforms take in data of the shape (NUM_CHANNELS, NUM_FEATURES)
# Although some have been written work on the last axis and may work on any-dimension data.
# Rewrite the transforms to accept the Package object and return a vector.


import scipy.io as sci
from scipy import stats
import numpy as np

import TimeAndFreq as Hill

def meanAndVarianceAndModeBySquibAbsoluteValues(package):
  newPacket = np.absolute(package.packet)
  return meanBySquib(newPacket) + varianceBySquib(newPacket) 

def thisIsRidiculous(package):
  """ For fun- sums squibs and then downsamples to 100Hz """
  return package.packet.sum(axis = 0)[::4].tolist()

def fft(package):
  packet = package.packet
  axis = packet.ndim - 1
  return np.fft.rfft(packet, axis=axis).real.tolist()

"""
Combines FFT with time and frequency correlation, taking both correlation coefficients and eigenvalues.
"""
def HillTFWTFC(package):  # collected call that calls the "Hill" FFTWithTimeFreqCorrelation
  return Hill.FFTWithTimeFreqCorrelation(package, 1, 48, 400, 'usf')

def HillTFWTFC_5Spawn(package):  # collected call that calls the "Hill" FFTWithTimeFreqCorrelation
  listOfFive = package.fiveSpawn()
  returnList = []
  for aPackage in listOfFive:
  	returnList.append(Hill.FFTWithTimeFreqCorrelation(aPackage, 1, 48, 400, 'usf'))

  return returnList

def deborahDoesTheAbsSumAndVarianceIncreaseWithTime(package):
  pass
  return returnList

def slopeAndRsquaredOfAbsSum(package):
  pass
  return returnList

def slopeAndRsquaredOfBinnedAbsVariance(package):
  pass
  return returnList
