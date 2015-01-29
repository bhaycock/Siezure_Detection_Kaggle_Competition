# Example quickrun to evaluate Epilepsy Data
# Making a standalone program that will read my data and play from there.
import epilepsyTools as eT
#import TestEstimators
import TransformsBH as transforms
import sys
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time

listOfDirectories = ['Dog_1'] #, 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
#listOfDirectories = ['Dog_5', 'Patient_1', 'Patient_2']
#listOfPretransforms = [makeFive, genIctals]
#listOfTransforms = []
#listOfClassifiers = []
#listOfPosttransforms [fiveAllVote]

# Note- the fiveAllVote and makeFive are a lot simplier than I thought... mainly just train on the small segments as is, and then we separate the test sets to make the votes in the predictions.
def main():
  finalList = []
  for stub in listOfDirectories:
    newList = []
    directory = "/Users/BaZ/Desktop/KaggleTest/Data/" + stub + "/"
    trainingList = eT.readDirectoryAndReturnTransformedTrainingList(directory, stub, transforms.thisIsRidiculous)
    
    # Write this out to the TransformedFile
    tic = time.time()
    filename = "./Output/" + stub + "Transformed_With_thisIsRidiculous.csv"
    outfile = open(filename, "w+")
    for line in trainingList:
      outfile.write(", ".join(str(line) + "\n"))
    outfile.close()
    print 'Write time for "open...close:" :', tic - time.time()

    tic = time.time()
    with open("./Output/AllDogsAndPatientsTransformed_With_thisIsRidiculous.csv", "a") as completeFile: # Lets try this with thing also. 
      for line in trainingList:
        completeFile.write(", ".join(str(line) + "\n"))
    print 'Write time for "with open:" :', tic - time.time()
    
    X_train = []
    y_train = []
    for line in trainingList:
      X_train.append(line[1:-1])
      y_train.append(line[-1])

    # Make array
    y_train = np.asarray(y_train)
    X_train = np.asarray(X_train)

    # Fit data
    print "Training"
    firstForest = RandomForestClassifier(n_estimators=3000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0)
    firstForest.fit(X_train, y_train)

    # Get test data w/ transform
    testList = eT.readDirectoryAndReturnTransformedList(directory, stub, 'test', transforms.thisIsRidiculous)

    for line in testList:
      newList.append([line[0], firstForest.predict_proba(line[1:])[0][1]])

    # Send to file
    print "Sending"
    filename = "./Output/" + stub + "ChristhisIsRidiculous_FirstForest_eT2.csv"
    outfile = open(filename, "w+")
    for line in newList:
      outfile.write(line[0] + ", " + str(line[1]) +"\n")
    outfile.close()
    print "Moving on"

    finalList = finalList + newList
  finalFile = open("./Output/ChristhisIsRidiculousfinalFile.csv", "w")
  for line in finalList:
    finalFile.write(line[0] + ", " + str(line[1]) + "\n")
  finalFile.close()
  return

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()
  