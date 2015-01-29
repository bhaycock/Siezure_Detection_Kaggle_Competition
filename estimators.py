#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
import os, sys
import csv
import numpy as np
import copy
"""
CURRENTLY IN PROGRESS 

Barry Haycock
2014 / 13 / 05
Kaggle competition Seizure prediction
https://www.kaggle.com/c/seizure-prediction/data

Set of methods to apply scikit.preprocesing and classifiers to
tranformed ictal data. Allows user to either metricise the 
effectiveness of one method, or to produce a submission for 
the Kaggle competition.

All of this will allow for fast generation of transform files,
which can be read in as a seperate step for training and predicting.
"""
# Need to implement some of the stuff in : 
# http://scikit-learn.org/stable/modules/cross_validation.html
# and
# http://scikit-learn.org/stable/modules/grid_search.html#grid-search 
# (especially grid-search)
from sklearn import metrics
from sklearn import cross_validation

# Potential classifiers:
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
# ExtraTrees, RandomForest, NiaveBayes, NeuralNetwork, Support Vector Machine, what else? 

# Keep in mind .feature_importances_
# and .cross_validation.cross_val_score
# and .roc_curve
# What was the method that returns the classifications and numbers?

# python's "yield"
# python's "with open(filename) as f"
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
"""classifiers = [RandomForestClassifier(n_estimators=3000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0),
      ExtraTreesClassifier (n_estimators=3000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0)
      KNeighborsClassifier(3),
      SVC(kernel="linear", C=0.025),
      SVC(gamma=2, C=1),
      DecisionTreeClassifier(max_depth=5),
      RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
      AdaBoostClassifier(),
      GaussianNB(),
      LDA(),
      QDA()]"""

classifiers = [RandomForestClassifier(n_estimators=3000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0)]

def return_roc_auc(name, dataDirectory, listOfScans, transform, classifier, KFold = False, preprocess= None, verbose = False, cv_ratio = 0.5, smartPreictal = True, fiveSplit = False, generatePreictal = False, generateInterictal = False):
  finalcv = []
  finalPredict = []
  for stub in listOfScans:
    newList = []
    directory =dataDirectory + stub + "/"
    trainingList = eT.readDirectoryAndReturnTransformedTrainingList(directory, stub, transform)

    clf = classifier
    if KFold: thisScore, this_y_cv, this_y_predict = kfold_evaluate_roc_auc(trainingList, clf, preprocess= preprocess, verbose = verbose, cv_ratio = cv_ratio, smartPreictal = smartPreictal, fiveSplit = fiveSplit, generatePreictal = generatePreictal, generateInterictal = generateInterictal)
    else: thisScore, this_y_cv, this_y_predict = cv_evaluate_roc_auc(trainingList, clf, preprocess= preprocess, verbose = verbose, cv_ratio = cv_ratio, smartPreictal = smartPreictal, fiveSplit = fiveSplit, generatePreictal = generatePreictal, generateInterictal = generateInterictal)
    
    if verbose:
      if KFold: print "Doing ", name, "on", stub, " yields a KFold- roc_auc of:", thisScore
      else: print "Doing ", name, "on", stub, " yields a roc_auc of:", thisScore, "when a cv ratio of ", cv_ratio, "is used." 
    finalcv += this_y_cv.tolist()
    finalPredict += this_y_predict.tolist()
  
  score = metrics.roc_auc_score(finalcv, finalPredict)
  if KFold: print name, " yields a KFold- roc_auc of:", score
  else: print name, " yields an roc_auc of:", score, "when a cv ratio of ", cv_ratio, "is used."
  print "Accuracy = ", accuracy(finalcv, finalPredict) 
  return (name, score)

def kfold_evaluate_roc_auc(trainData, classifier, preprocess= None, verbose = False, cv_ratio = 0.5, smartPreictal = True, fiveSplit = False, generatePreictal = False, generateInterictal = False):
  """ Method that applies the preprocessing step to transformed training data and then splits the 
  data via the cv ratio before calculating the roc_auc. The smartPreictal option forces the cv set to contain at least
  one full preictal scan (not implemented for KFold). The verbose flag prints a report to the screen about this run In progress. 
  """
  # Get X and y from the trainData:
  X = [record[1:-1] for record in trainData]
  yDict = { 'preictal' : 1,'interictal':0}
  try:
    y = [yDict[record[-1]] for record in trainData]
  except KeyError:
    dumpFile = open("dumpfile.dat", "w")
    for record in trainData:
      dumpFile.write(str(record) + "\n")
  # Preprocess if neccessary
  if preprocess:
    X = preprocess(X)
  X = np.asarray(X)
  y = np.asarray(y)

  # Split data
  # Issue here is that I'm unsure if the preictal data needs to be sensibly grabbed for the CV, 
  #                   gonna assume it doesn't and I can add a new argument to the method call later.
  kf = cross_validation.KFold(len(y),n_folds=5,shuffle=True)
  y_pred = copy.copy(y)

  # Iterate through folds
  for train_index, test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    y_train = y[train_index]
    print "y.shape, y_pred.shape, y_train.shape", y.shape, y_pred.shape, y_train.shape
    print "len(y_train), len(X_train)", len(y_train), len(X_train) 
    
    # Initialize a classifier with key word arguments
    clf = classifier    # May have to fix this with a complete re-ititialize
    print "X_train.shape, y_train.shape", X_train.shape, y_train.shape
    clf.fit(X_train,y_train)
    # Predict this stuff
    print "train_index", train_index
    print "test_index", test_index
    print "len(X_test)", len(X_test)
    y_predicted = clf.predict_proba(X_test)
    #y_pred[test_index] = clf.predict_proba(X_test)

  # return report if requested
  # I can plot the ROC for showing the team in here, and I can add a few bits in the other metrics.
  # I think I wanna try that GRID thing also, though. It should give me an optimized set of params for the set.
  if verbose:
    pass
    # In the verbose, I think I'd like to see something about predictive power (although feature reduction can be automated) 
    # and something like maybe the sensitivity and specitivity. Maybe the plot of the ROC.

  
  # return with result
  print "Is there an errror here? len(y_pred[test_index])", len(y_pred[test_index])
  return (metrics.roc_auc_score(y_pred[test_index], y_predicted[:,0]), y_pred[test_index], y_predicted)



def cv_evaluate_roc_auc(trainData, classifier, preprocess= None, verbose = False, cv_ratio = 0.5, smartPreictal = True, fiveSplit = False, generatePreictal = False, generateInterictal = False):
  """ Method that applies the preprocessing step to transformed training data and then splits the 
	data via the cv ratio before calculating the roc_auc. The smartPreictal option forces the cv set to contain at least
	one full preictal scan. The verbose flag prints a report to the screen about this run. 
  """
	# Get X and y from the trainData:
#  X = trainData[:,1:-1]
#  y = trainData[:, -1]
  
  # Preprocess if neccessary
  if preprocess:
    X = preprocess(X)

  # Split data
  # Issue here is that I'm unsure if the preictal data needs to be sensibly grabbed for the CV, 
  #                   gonna assume it doesn't and I can add a new argument to the method call later.
  #X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, y, test_size=cv_ratio)
  if smartPreictal:
    X_preictal = [record[1:-1] for record in trainData if record[-1] == 'preictal']
    y_preictal = [1 for record in trainData if record[-1] == 'preictal']
    X_interictal = [record[1:-1] for record in trainData if record[-1] == 'interictal']
    y_interictal  = [0 for record in trainData if record[-1] == 'interictal']

    X_train_preictal, X_cv_preictal, y_train_preictal, y_cv_preictal = cross_validation.train_test_split(X_preictal, y_preictal, test_size=cv_ratio)
    X_train_interictal, X_cv_interictal, y_train_interictal, y_cv_interictal = cross_validation.train_test_split(X_interictal, y_interictal, test_size=cv_ratio) 

    X_train = np.concatenate((X_train_preictal, X_train_interictal), axis = 0)
    y_train = np.concatenate((y_train_preictal, y_train_interictal), axis = 0)

    X_cv = np.concatenate((X_cv_preictal, X_cv_interictal), axis = 0)
    y_cv = np.concatenate((y_cv_preictal, y_cv_interictal), axis = 0)
  else:
   # Get X and y from the trainData:
    X = [record[1:-1] for record in trainData]
    yDict = { 'preictal' : 1,'interictal':0}
    y = [yDict[record[-1]] for record in trainData] 
    X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, y, test_size=cv_ratio)  

  # Apply classifier
  clf = classifier.fit(X_train, y_train)

  # Predict this stuff
  y_predicted = clf.predict(X_cv)

  # return report if requested
  # I can plot the ROC for showing the team in here, and I can add a few bits in the other metrics.
  # I think I wanna try that GRID thing also, though. It should give me an optimized set of params for the set.
  if verbose:
    pass
    # In the verbose, I think I'd like to see something about predictive power (although feature reduction can be automated) 
    # and something like maybe the sensitivity and specitivity. Maybe the plot of the ROC.

  
  # return with result

  return (metrics.roc_auc_score(y_cv, y_predicted), y_cv, y_predicted)

def predict(trainData, testData, classifier, preprocess = None, smartPreictal = True):
  """ Method to predict based on the trainign data, optional preprocessing step and the chosen classifier. Returns a Kaggle-submittable list.
  """
  # Get X and y from the trainData:
  X_train = trainData[:,1:-1]
  y_train = trainData[:, -1]

  # Get X from the testData:
  X_test = testData[:,1:-1]
  
  # Preprocess if neccessary
  if preprocess:
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

  # Apply classifier
  clf = classifier.fit(X_train, y_train)  

  # Predict
  y_predicted = clf.predict(X_test)
  
  return predictions

def readCSVandGetROC_AUC(filename, classifier, preprocess= None, verbose = False, cv_ratio = 0.5, smartPreictal = True):
  """ Quick routine to read in a .csv in the "name, X[], y] format and pass it into the evaluate_roc_auc method 
  """
  trainData = []
  with open(filename, "r") as transformedFile:
#    trainData = transformedFile.readlines()
    trainCSV = csv.reader(transformedFile)
    for row in trainCSV:
      rowData = []
      for cell in row:
        try:
          rowData.append(float(cell))
        except ValueError:
          if str(cell) == " 'interictal'": rowData.append(str('interictal'))
          elif str(cell) == " 'preictal'": rowData.append(str('preictal'))
          else: rowData.append(str(cell))

      trainData.append(rowData)

  return evaluate_roc_auc(trainData, classifier, preprocess, verbose, cv_ratio, smartPreictal)

def accuracy(y_true,y_pred):
    # NumPy interpretes True and False as 1. and 0.
    return np.mean(y_true == y_pred)

