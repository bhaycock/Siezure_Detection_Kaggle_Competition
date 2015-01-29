#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
# Example quickrun to evaluate ROC_AUC of a pipeline
# Making a standalone program that will read data and
# apply the "thisIsRidiculous" transform then output
# the ROC.
import epilepsyTools as eT
#import TestEstimators
import TransformsBH as transforms
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
import time
import estimators as em
#import transform_Hills_cgv2B as Hill

listOfDirectories = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
#listOfDirectories = ['Dog_5', 'Patient_1', 'Patient_2']
#listOfPretransforms = [makeFive, genIctals]
#listOfTransforms = []
#listOfClassifiers = []
#listOfPosttransforms [fiveAllVote]


# Note- the fiveAllVote and makeFive are a lot simplier than I thought... mainly just train on the small segments as is, and then we separate the test sets to make the votes in the predictions.
def main():
  finalcv = []
  finalPredict = []
  for stub in listOfDirectories:
    newList = []
    directory = "/Users/BaZ/Desktop/KaggleTest/Data/" + stub + "/"
    trainingList = eT.readDirectoryAndReturnTransformedTrainingList(directory, stub, transforms.fft)

    # Initialize classifier:
    firstForest = RandomForestClassifier(n_estimators=3000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0)
    thisScore, this_y_cv, this_y_predict = em.kfold_evaluate_roc_auc(trainingList, firstForest)
    
    print "KF- Doing RandomForest on transforms.fft on", stub, " yields a roc_auc of:", thisScore
    finalcv += this_y_cv.tolist()
    finalPredict += this_y_predict.tolist()

  print "KF- Over all the runs in the list, using RF_3000_mss1_bsF_nj4_rs0 using transforms.fft, roc_auc = ", metrics.roc_auc_score(finalcv, finalPredict)
  print "Accuracy = ", em.accuracy(finalcv, finalPredict) 
  return

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()