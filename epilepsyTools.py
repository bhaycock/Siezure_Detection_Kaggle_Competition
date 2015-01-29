#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
import scipy.io as sci
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re
import copy
from time import time
import TransformsBH as transforms 
"""
Barry Haycock & Christophe Guerrier
2014 / 13 / 05
Kaggle competition Seizure prediction
https://www.kaggle.com/c/seizure-prediction/data
rewritten with a new EEGpackage class that will 
allow for immutables such as the kind of package 
it is (preictal, interictal, test) and allow a .applyTransform() 
method and to combine two packets into a child packet and 
to spawn five sub-packages from a parent.

All of this will allow for fast generation of transform files,
which can be read in as a seperate step for training and predicting.
"""

class EEGpackage(object):
  def __init__(self, matfileData, name, fileKind):
    self.packet = matfileData[0][0][0]
    self.duration = matfileData[0][0][1][0][0]
    self.frequency = matfileData[0][0][2][0][0]
    self.probeNames =[]
    for nameAsNumpy in matfileData[0][0][3][0]:
      self.probeNames.append(nameAsNumpy[0])
    try: 
      self.index = matfileData[0][0][4][0][0]   # Test data sets do not include an index.
    except IndexError:
      pass
    self.name = name
    # Set up what kind of data package this is.
    self.__preictal = False
    self.__interictal = False
    self.__test = False
    if fileKind == 'preictal': self.__preictal = True
    elif fileKind == 'interictal': self.__interictal = True
    elif fileKind == 'test': self.__test = True
    else: print "Unrecognised package type (preictal, interictal, test), this is probably bad." 
    return
  
  @property
  def fileKind(self):
    if self.__preictal == True: return 'preictal'
    elif self.__interictal == True: return 'interictal'
    elif self.__test == True: return 'test'
    else: return 'Unrecognised Kind'

  @property
  def preictal(self):
    return self.__preictal

  @property
  def interictal(self):
    return self.__interictal
  
  @property
  def test(self):
    return self.__test
      
  #Helper functions
  def applyTransform(self, transform):
    """ object equivilent of transform(EEGpackage) """
    return transform(self)

  def plotPacket(self, endval = -1, startval = 0):
    """ Chucks out a rough and ready plot of EEG Squibs """
    EEGmax = self.packet.max()
    for isquib in range(len(self.packet)):
      yVals = (isquib * (1000)) + self.packet[isquib, startval:endval]
      xVals = xrange(len(self.packet[isquib, startval:endval]))
      plt.plot(xVals, yVals, '-r')
    plt.ylabel('EEG')
    plt.xlabel('point number')
    #plt.axis('tight')
    plt.show()
    return
  
  def fiveSpawn(self):
    """ returns a list of five 'new' EEGpackages, each containing 2 minute 
    sections of the squibs in the original
    Produces objects with name altered to have _1 through _5

    Looses up to 4 data points in order to make the array divisible by 5. I wouldn't worry about that."""
    returnList = []
    try:  # Try splitting the packet into 5
      packetList = np.hsplit(self.packet, 5)
    except ValueError: # If that doesn't work, try splitting the packet with one less column into five
      try:
        packetList = np.hsplit(self.packet[:,:-1],5)
      except ValueError: # If that doesn't work, try splitting the packet with two less columns into five
        try:
          packetList = np.hsplit(self.packet[:,:-2],5)
        except ValueError: # If that doesn't work, try splitting the packet with three less columns into five
          try:
            packetList = np.hsplit(self.packet[:,:-3],5)
          except ValueError: # If that doesn't work, try splitting the packet with four less columns into five
            try:
              packetList = np.hsplit(self.packet[:,:-4],5)
            except ValueError:
              print "This packet doesn't split into five"
              return

    numPoints = self.packet.shape[1]
    numPointsPerSub = int(numPoints / 5)
    print "fiveSpawn!"
    for i in range(5):
      returnList.append(copy.copy(self))
      returnList[i].packet = packetList[i]
      returnList[i].name = self.name + "_" + str(i)
    return returnList

  def combineToNew(self, other):
    """returns the 'child' of self and other after confirming that these two can be combined
    the child is defined as the second half of the first packages' packet (self) and the first half of the other package"""
    #First confirm that these two packages are compatible:
    if self.test == True or other.test == True: 
      print "You're trying to combine a test package, that doesn't work"
      return
    #elif self.compatibleForCombine(other):
    elif self.duration != other.duration or self.frequency != other.frequency or self.probeNames != other.probeNames or self.fileKind != other.fileKind or self.packet.shape != other.packet.shape:
      print "These are not compatible packages for combination by compatibleForCombine"
      return
    elif self.index +1 != other.index :
      print "These are not compatible by index"
      return

    # Copy self
    returnPackage = copy.copy(self)
    # Create new packet
    numCols = self.packet.shape[1]
    returnPackage.packet = np.concatenate((self.packet[:,numCols:], other.packet[:,:numCols+1]), axis = 1)
    # Create new index number
    returnPackage.index = float(self.index + 0.5)

    # Create new name
    returnPackage.name = "half" + self.name + "+half" + other.name 

    # Return the child packet
    return returnPackage

  def compatibleForCombine(self, other):
    """ Helper method to assist with combineToNew, compares all internal variables except the packet and the index. Checks filename."""
    if self.duration != other.duration or self.frequency != other.frequency or self.probeNames != other.probeNames or self.fileKind != other.fileKind or self.packet.shape != other.packet.shape: return False
    if self.index +1 != other.index : return False
    return True

  def downsample(self, divisor):
    """ downsamples the packet by divisor, so if you have a 400Hz packet, and call EEGpackage.downsample(2), you will now have a 200Hz packet.
    this routine requires an integer divisor and updates the .frequency in the object."""
    self.packet = self.packet[:, ::divisor]
    self.frequency = self.frequency / divisor
    return

def readMatReturnPackage(filename, transformer = ""):
  """ Accepts a matlab filename, returns the EEG package with data transformed by the transformer, if applicable 
  Automagically works out the file kind (preictal, interictal, test) from the data file names in the .mat file"""
  matlabFile = sci.loadmat(filename)
  listOfKeys = matlabFile.keys()
  listOfKeys.remove('__version__')
  listOfKeys.remove('__header__')
  listOfKeys.remove('__globals__')
  segment = listOfKeys[0]
  fileKind = re.match(r'[a-z]+', segment).group()

  if transformer == "": 
  	return EEGpackage(matlabFile[segment], filename, fileKind)
  else:
  	return EEGpackage(matlabFile[segment], filename, fileKind)
  	data = transformer(returnPackage.packet)
  	del returnPackage.packet
  	returnPackage.data = data
  	return returnPackage


def readDirectoryAndReturnTransformedList(directory, stub, component, transform, fiveSpawn=False):
  """ Version 0.1 had a "readTrainingDirectoryAndReturnPackageList", this is insane on memory usage when all I need
  is a list of lists. Where column0 is the filename, columns 1:[-1] are the X features and column[-1] is the dependent variable

  Until I think of a better way, this subroutine will take the directory, the stub, the component and the transform function and return
  the list of lists of transformed data. The "better way" will work for training and then test- probably by a "read training method", which will
  call this twice and a read test method.
  """
  returnList = []
  done = False
  i = 0
  while not done:
    i += 1
    curPackListEntry = []
    DirectoryFilename = '%s/%s_%s_segment_%04d.mat' % (directory, stub, component, i)
    print "Reading in File:", DirectoryFilename
    if os.path.exists(DirectoryFilename):
      curPackage = readMatReturnPackage(DirectoryFilename)
      filename = '%s_%s_segment_%04d.mat' % (stub, component, i)
      if fiveSpawn: ListOfPackages = curPackage.fiveSpawn()
      else: ListOfPackages = [curPackage]
      for curPackage5 in ListOfPackages:
        curPackListEntry = []
        if fiveSpawn: curPackListEntry.append((str(filename) + str(curPackage5.name[-2:])))
        else: curPackListEntry.append(filename)
        print "Transform and append"
        curPackListEntry = curPackListEntry + curPackage5.applyTransform(transform)
        if not curPackage5.test: curPackListEntry.append(curPackage5.fileKind)
        else: print "TestFile"
        #print curPackListEntry
        returnList.append(curPackListEntry)
    else:
      if i == 1:
        raise Exception("file %s not found" % DirectoryFilename)
      done = True

  return returnList

def readDirectoryAndReturnTransformedTrainingList(directory, stub, transform, fiveSpawn=False):
  """ Combines above to return all training data in a directory """
  returnList = readDirectoryAndReturnTransformedList(directory, stub, 'interictal', transform, fiveSpawn)
  returnList = returnList + readDirectoryAndReturnTransformedList(directory, stub, 'preictal', transform, fiveSpawn)
  return returnList

def pretendTransform(package):
  """ Just returns a list of ten 1's"""
  return [1]*10

def pretendFail(package):
  return (1/0)

def passthroughTransform(package):
  returnList = []
  for iline in package.packet:
    returnList += iline.tolist()
  return returnList
  
def testTransform(transform, tryPlot = True, fiveSpawn=False):
  """ Tests your transform by reading in some sample ictal .mat files as generated by P.Collins and 
  applying this code. If the result is plotable, it will call matplotlib and show you the result of the transform. 
  It returns a list of the resultant feature vectors of the tests.
  """
  tic = time()
  returnList = []
  try:
    returnList = readDirectoryAndReturnTransformedList("./testMATfiles", "collins", "simulated", transform, fiveSpawn)
    if fiveSpawn:
      if type(returnList[0]) is list:
        print "Error: returning a list of lists in transform"
        return
    elif type(returnList[0]) is list:
      print "Error: returning a list of lists in transform"
      return
  except Exception:
    print "There was an error in reading in the files or failure in the transform:"
    print sys.exc_info()[1]
    print sys.exc_info()[2]
    return
  timeTook = time() - tic
  print "Read Directory And Return Transformed List Took", timeTook,  "seconds "
  print "         This corresponds to a time of ", timeTook / 6 , "per read-and-transform operation."  
  # Alter packet names based on what I know is actually in here.
  returnList[0][0] = "Ones"
  returnList[1][0] = "Zeros"
  returnList[2][0] = "Random Noise"
  returnList[3][0] = "Clean Sine 50Hz"
  returnList[4][0] = "50Hz Sine with Noise"
  returnList[5][0] = "50Hz Sine with 10Hz Sine with Noise"

  
  # Attempt to plot the 4 feature vectors:
  def plotThisTransformTest(returnList):
    for line in returnList:
      plt.plot(line[1:-2], label = line[0])
  
    plt.legend()
    plt.xlabel('point number')
    #plt.axis('tight')
    plt.show()
    return

  if tryPlot: 
    try: 
      plotThisTransformTest(returnList)
    except:
      print "Error in plotting  this data:"
      print sys.exc_info()[1]
      print sys.exc_info()[2]

  # Return the lists.
  return returnList

def return_roc_auc(name, dataDirectory, listOfScans, transform, classifier, KFold = False, preprocess= None, verbose = False, cv_ratio = 0.5, smartPreictal = True, fiveSplit = False, generatePreictal = False, generateInterictal = False):
  """ Outputs Roc_Auc score """
  finalcv = []
  finalPredict = []
  for stub in listOfScans:
    newList = []
    directory = dataDirectory + stub + "/"
    trainingList = readDirectoryAndReturnTransformedTrainingList(directory, stub, transform)
    if KFold: thisScore, this_y_cv, this_y_predict = kfold_evaluate_roc_auc(trainingList, clf, preprocess= preprocess, verbose = verbose, cv_ratio = cv_ratio, smartPreictal = smartPreictal, fiveSplit = fiveSplit, generatePreictal = generatePreictal, generateInterictal = generateInterictal)
    else: thisScore, this_y_cv, this_y_predict = cv_evaluate_roc_auc(trainingList, clf, preprocess= preprocess, verbose = verbose, cv_ratio = cv_ratio, smartPreictal = smartPreictal, fiveSplit = fiveSplit, generatePreictal = generatePreictal, generateInterictal = generateInterictal)
    
    if verbose:
      if KFold: print "Doing ", name, "on", stub, " yields a KFold- roc_auc of:", thisScore
      else: print "Doing ", name, "on", stub, " yields a roc_auc of:", thisScore, "when a cv ratio of ", cv_ratio, "is used." 
      print "Calling ShowFeatureImportances at end"
    finalcv += this_y_cv.tolist()
    finalPredict += this_y_predict.tolist()
  
  score = metrics.roc_auc_score(finalcv, finalPredict)
  if KFold: print name, " yields a KFold- roc_auc of:", score
  else: print name, " yields an roc_auc of:", score, "when a cv ratio of ", cv_ratio, "is used."
  print "Accuracy = ", accuracy(finalcv, finalPredict)
   
  if verbose: print showFeatureImportances(clf)
  return score

def showFeatureImportances(clf, plot = True):
  importances = clf.feature_importances_
  try:
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
  except:
    std = 0.0
  indices = np.argsort(importances)[::-1]

  # Print the feature ranking
  print("Feature ranking:")

  for f in range(10):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

  # Plot the feature importances of the forest
  if plotThem:
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(10), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(10), indices)
    plt.xlim([-1, 10])
    plt.show()
  return importances

def finalPredict(name, dataDirectory, listOfScans, transform, clf, preprocess= None, fiveSplit = False, returnPred = False):
  """ Reads in, carrys out predictions and posts to a csv file.
  needs work.
  """
  finalPredictionsList =[]
  # Loop over scans
  for stub in listOfScans:
    # Read training data
    directory =dataDirectory + stub + "/"
    print "opening training", directory
    trainingList = readDirectoryAndReturnTransformedTrainingList(directory, stub, transform, fiveSpawn = fiveSplit)

    # Read test data
    directory =dataDirectory + stub + "/"
    print "opening test", directory
    testList = readDirectoryAndReturnTransformedList(directory, stub, 'test', transform, fiveSpawn = fiveSplit)
    
    # Train and return results
    print "calling predict", stub
    predictionsList = predictThis(trainingList, testList, clf, preprocess= preprocess, fiveSplit = fiveSplit)

    # Append results to final list.
    print "writing to FList"
    finalPredictionsList += predictionsList
  print "len(finalPredictionsList)", len(finalPredictionsList)
  #print finalPredictionsList
  # Write out to file
  #filename1 = name + "_predictions1.csv"
  #finalFile1 = open(filename1, "w")
  filename2 = name + "_predictions2.csv"
  finalFile2 = open(filename2, "w")
  print finalPredictionsList

  for line in finalPredictionsList:
    #finalFile1.write(line[0] + ", " + str(line[1][0]) + "\n")
    finalFile2.write(line[0] + ", " + str(line[1][1]) + "\n")
  #finalFile1.close()
  finalFile2.close()
  if returnPred : return finalPredictionsList
  return 

def predictThis(trainData, testData, clf, preprocess= None, fiveSplit = False):
  """ Carries out a predition for a sepcific scan / classifier set.
  """
    # Get X_train and y_train from the trainData:
  print "Entering predictThis"
  X_train = [record[1:-1] for record in trainData]
  yDict = { 'preictal' : 1,'interictal':0}
  y_train = [yDict[record[-1]] for record in trainData]
  print "Train defined"
  # Preprocess if neccessary
  if preprocess:
    X_train = preprocess(X_train)
  X_train = np.asarray(X_train)
  y_train = np.asarray(y_train)
  print "preprocess Preprocessed"
  # Get X_test and names from the testData:
  X_test = [record[1:] for record in testData]
  filenames = [record[0] for record in testData]
  print "Test Defined"
  # Preprocess if neccessary
  if preprocess:
    X_test = preprocess(X_test)
  X_test = np.asarray(X_test)
  print "Test Preprocessed, calling classifier"
# Apply classifier
  classifier = clf
  classifier.fit(X_train, y_train)  
  print "fitting fitted"
  #Predict away
  print "predictions predicting"
  predictions = classifier.predict_proba(X_test)
  print "Homeward going"
  if fiveSplit:
    return fiveSplitBackToUnsplit(zip(filenames, predictions), returnLists = True)
  else:
    return (zip(filenames, predictions))



def fiveSplitBackToUnsplit(filenamePredictionList, returnLists = False, returnMaskedMean = False, returnMeanAndVar = False, doesThePredictionIncrease = False):
  """ Gets the zipped list of zip(filenames, predictions) where predictions are what prediction is made and 
  filename is the filename as modified by the "fivesplit" and puts everything back together."""
  print "5Back", returnLists
  # I can address by the name being constructed of (filename_NUM), therefore filename = filename[:-2] and index = filename[-1]
  returnList = []
  prevname = ""
  for filename, prediction in filenamePredictionList:

    if filename[:-2] == prevname:
      print "prevname =", prevname
      listOfFivePredictions.append(prediction)
        #Check this data has been split, otherwise fail out loudly
      try: 
        if int(filename[-1])>5: 
          print "Your fiveSplit is greater than 5. This is an error."
          return
        elif int(filename[-1])<1:
          print "Your fiveSplit is less than zero. This is an error."
          return
      except ValueError:
        print "FiveSplit has been called on non-indexed packages, this is probaby an error."
        return  
      if int(filename[-1])==5:
        returnList.append(prevname, listOfFivePredictions)
    else:
      prevname = filename[:-2]
      try:
        if int(filename[-1]) != '1':
          print "Your First fiveSplit is not indexed as 1. This is an error.", filename
          return
      except ValueError:
        print "FiveSplit has been called on non-indexed packages, this is probaby an error."
        return

      listOfFivePredictions = []
      listOfFivePredictions.append(prediction)
  if returnLists: return returnList 
  
  if returnMeanAndVar:
    finalList = []
    for filename, predList in returnList:
      finalList.append((filename, np.mean(predList), np.var(predList)))
    return finalList
  
  if returnMaskedMean:
    finalList = []
    for filename, predList in returnList:
      weighMask = [1.0 if x >= 0.5 else 0.0 for x in predList]
      if sum(weighMask) < 3.0 :
        #Less of these guys call interictal rather than preictal.
        maskedMean = (sum([predList[i] * (1-weighMask[i]) for i in range(5)]))/5
      else:
        #More sub packages think they're preicatal:
        maskedMean = (sum([predList[i] * weighMask[i] for i in range(5)]))/5
      finalList.append(filename, maskedMean)
    return finalList
  
  if doesThePredictionIncrease:
    finalList = []
    for filename, predList in returnList:
      if predList[0] <= predList[1] and predList[1] <= predList[2] and predList[2] <= predList[3] and predList[3] <= predList[4]: 
        finalList.append(filename, 1.0)
      else: finalList.append(filename, 0.0)
    return finalList
  
  if doesThePredictionIncreaseSum:
    returnDoesThePredSum = []
    for filename, predList in returnList:
      generallyIncrease = 0.0
      for i in range(1,4):
        if predList[i-1] <= predList[i]: generallyIncrease += 0.25 
      doesThePredictionIncreaseSum.append(generallyIncrease)
    return zip(filenames, doesThePredictionIncreaseSum)


  if returnOnlyLast:
    finalList = []
    for filename, predList in returnList:
      finalList.append(filename, predList[-1])
    return finalList










