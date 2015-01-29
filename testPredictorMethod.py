# Example quickrun to evaluate Epilepsy Data
# Making a standalone program that will read my data and play from there.
import epilepsyTools as eT
#import TestEstimators
import TransformsBH as transforms
import sys
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
#import transform_Hills_cgv2B as Hill as Hill

listOfScans = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
#listOfDirectories = ['Dog_5', 'Patient_1', 'Patient_2']
#listOfPretransforms = [makeFive, genIctals]
#listOfTransforms = []
#listOfClassifiers = []
#listOfPosttransforms [fiveAllVote]

# Note- the fiveAllVote and makeFive are a lot simplier than I thought... mainly just train on the small segments as is, and then we separate the test sets to make the votes in the predictions.
def main():
  directory = "/Users/BaZ/Desktop/KaggleTest/Data/"

  name = "checkFFT_runPassThrough"
  clf2 = RandomForestClassifier(n_estimators=30, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0)
  eT.finalPredict(name, directory, listOfScans, eT.passthroughTransform, clf2)

  name = "checkFFT"
  #clf2 = RandomForestClassifier(n_estimators=30, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0)
  #eT.finalPredict(name, directory, listOfScans, transforms.fft, clf2)

  name = "test1_ridiculous_All"
#  clf = RandomForestClassifier(n_estimators=3000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0)
#  eT.finalPredict(name, directory, listOfScans, transforms.thisIsRidiculous, clf)


  return

if __name__ == '__main__':
  main()
