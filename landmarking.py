from collections import OrderedDict
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import ensemble, svm, linear_model

import utils
from utils import stopwatch

def cross_validate_classifier(clf, X, y, folds):
	"""
	Performs a cross-validation experiment for the classifier clf,
	reporting mean accuracy and time to run (as well as total time,
	but this was later left out in the study).
	"""

	accuracies = []
	times = []

	for train, test in folds:
		with stopwatch() as sw:
			clf.fit(X[train], y[train])
			accuracy = clf.score(X[test], y[test])
			accuracies.append(accuracy)
		times.append(sw.duration)

	return np.mean(accuracies), sum(times), np.mean(times)


def landmarking_metafeature_names():
	return ["DecisionTreeGiniDepth1Accuracy", "DecisionTreeGiniDepth1TimeSum", "DecisionTreeGiniDepth1TimeMean",
			"DecisionTreeEntropyDepth1Accuracy", "DecisionTreeEntropyDepth1TimeSum", "DecisionTreeEntropyDepth1TimeMean",
			"DecisionTreeGiniDepth2Accuracy", "DecisionTreeGiniDepth2TimeSum", "DecisionTreeGiniDepth2TimeMean",
			"DecisionTreeEntropyDepth2Accuracy", "DecisionTreeEntropyDepth2TimeSum", "DecisionTreeEntropyDepth2TimeMean",
			"DecisionTreeGiniDepth4Accuracy", "DecisionTreeGiniDepth4TimeSum", "DecisionTreeGiniDepth4TimeMean",
			"DecisionTreeEntropyDepth4Accuracy", "DecisionTreeEntropyDepth4TimeSum", "DecisionTreeEntropyDepth4TimeMean",
			"DecisionTreeGiniAccuracy", "DecisionTreeGiniTimeSum", "DecisionTreeGiniTimeMean",
			"GaussianNBAccuracy", "GaussianNBTimeSum", "GaussianNBTimeMean",
			"1NNAccuracy", "1NNTimeSum", "1NNTimeMean"]

def landmarker_metafeatures(X, y, categorical, folds):
	utils.input_check(X, y, categorical)
	features = OrderedDict()

	# TODO: When having folds, do cross-validation instead as it takes more time
	# and also can give a fair indication of predictive accuracy
	for i in [1,2,4]:
		accuracy, total_time, mean_time = cross_validate_classifier(DecisionTreeClassifier(criterion='gini', max_depth=i), X, y, folds)
		features["DecisionTreeGiniDepth{}Accuracy".format(i)] = accuracy
		features["DecisionTreeGiniDepth{}TimeSum".format(i)] = total_time
		features["DecisionTreeGiniDepth{}TimeMean".format(i)] = mean_time

		accuracy, total_time, mean_time = cross_validate_classifier(DecisionTreeClassifier(criterion='entropy', max_depth=i), X, y, folds)
		features["DecisionTreeEntropyDepth{}Accuracy".format(i)] = accuracy
		features["DecisionTreeEntropyDepth{}TimeSum".format(i)] = total_time
		features["DecisionTreeEntropyDepth{}TimeMean".format(i)] = mean_time

	accuracy, total_time, mean_time = cross_validate_classifier(DecisionTreeClassifier(), X, y, folds)
	features["DecisionTreeGiniAccuracy".format(i)] = accuracy
	features["DecisionTreeGiniTimeSum".format(i)] = total_time
	features["DecisionTreeGiniTimeMean".format(i)] = mean_time

	accuracy, total_time, mean_time = cross_validate_classifier(GaussianNB(), X, y, folds)
	features["GaussianNBAccuracy".format(i)] = accuracy
	features["GaussianNBTimeSum".format(i)] = total_time
	features["GaussianNBTimeMean".format(i)] = mean_time

	accuracy, total_time, mean_time = cross_validate_classifier(KNeighborsClassifier(n_neighbors=1), X, y, folds)
	features["1NNAccuracy".format(i)] = accuracy
	features["1NNTimeSum".format(i)] = total_time
	features["1NNTimeMean".format(i)] = mean_time
	return features

def subsample_metafeature_names():
	return ["SubsampleRandomForestAccuracy", "SubsampleRandomForestMeanTime",
			"SubsampleSVCAccuracy", "SubsampleSVCMeanTime",
			"SubsampleBoostingAccuracy", "SubsampleBoostingMeanTime"]

def subsample_metafeatures(X, y, categorical, folds):
	utils.input_check(X, y, categorical)
	features = OrderedDict()

	accuracy, total_time, mean_time = cross_validate_classifier(ensemble.RandomForestClassifier(), X, y, folds)
	features["SubsampleRandomForestAccuracy"] = accuracy
	features["SubsampleRandomForestMeanTime"] = mean_time

	accuracy, total_time, mean_time = cross_validate_classifier(svm.SVC(), X, y, folds)
	features["SubsampleSVCAccuracy"] = accuracy
	features["SubsampleSVCMeanTime"] = mean_time

	accuracy, total_time, mean_time = cross_validate_classifier(ensemble.GradientBoostingClassifier(), X, y, folds)
	features["SubsampleBoostingAccuracy"] = accuracy
	features["SubsampleBoostingMeanTime"] = mean_time

	return features
