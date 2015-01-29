"""
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

def evaluate_roc_auc(trainData, classifier, preprocess= None, verbose = False, cv_ratio = 0.5, smartPreictal = True):
	""" Method that applies the preprocessing step to transformed training data and then splits the 
	data via the cv ratio before calculating the roc_auc. The smartPreictal option forces the cv set to contain at least
	one full preictal scan. The verbose flag prints a report to the screen about this run. 
	"""
	# Get X and y from the trainData:
  X = trainData[:,1:-1]
  y = trainData[:, -1]
  
  # Preprocess if neccessary
  if preprocess:
    X = preprocess(X)

  # Split data
  # Issue here is that I'm unsure if the preictal data needs to be sensibly grabbed for the CV, 
  #                   gonna assume it doesn't and I can add a new argument to the method call later.
  X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, y, test_size=cv_ratio)

  # Apply classifier


  # return report if requested

  # return with result

	return roc_auc

def predict(trainData, testData, classifier, preprocess = None):
	""" Method to predict based on the trainign data, optional preprocessing step and the chosen classifier. Returns a Kaggle-submittable list.
  """
	pass
	return predictions

