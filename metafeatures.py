from __future__ import division
from collections import OrderedDict
import numpy as np
import scipy.stats
import sklearn.metrics
from landmarking import landmarker_metafeatures
import utils

def simple_metafeature_names():
	return ["NumberOfInstances", "LogNumberOfInstances", "NumberOfFeatures", "LogNumberOfFeatures", "DatasetRatio",
			"LogDatasetRatio", "InverseDatasetRatio", "LogInverseDatasetRatio", "NumberOfClasses", "NumberOfCategoricalFeatures",
			"NumberOfNumericFeatures", "RatioNumericalToNominal", "RatioNominalToNumerical", "ClassProbabilityMin",
			"ClassProbabilityMax", "ClassProbabilityMean", "ClassProbabilitySTD", 
			"SymbolsMin", "SymbolsMax", "SymbolsMean", "SymbolsSTD", "SymbolsSum", "SimpleFeatureTime"]

def simple_metafeatures(X, y, categorical):
	utils.input_check(X, y, categorical)
	features = OrderedDict()
	n = X.shape[0]
	p = X.shape[1]

	with utils.stopwatch() as sw:
		features["NumberOfInstances"] = n
		features["LogNumberOfInstances"] = np.log(n)
		features["NumberOfFeatures"] = p
		features["LogNumberOfFeatures"] = np.log(p)
		features["DatasetRatio"] = p / n
		features["LogDatasetRatio"] = np.log(p / n)
		features["InverseDatasetRatio"] = n / p 
		features["LogInverseDatasetRatio"] = np.log(n / p)

		classes, counts = np.unique(y, return_counts = True)
		nrNominal = sum(categorical)
		nrNumeric = len(categorical)-sum(categorical)

		features["NumberOfClasses"] = classes.shape[0]
		features["NumberOfCategoricalFeatures"] = nrNominal
		features["NumberOfNumericFeatures"] = nrNumeric

		features["RatioNumericalToNominal"] = nrNumeric / nrNominal if nrNominal > 0 else 0
		features["RatioNominalToNumerical"] = nrNominal / nrNumeric if nrNumeric > 0 else 0

		class_probabilities = [ count / n for count in counts ]
		features["ClassProbabilityMin"] = np.min(class_probabilities)
		features["ClassProbabilityMax"] = np.max(class_probabilities)
		features["ClassProbabilityMean"] = np.mean(class_probabilities)
		features["ClassProbabilitySTD"] = np.std(class_probabilities)

		symbols_per_column = [ np.unique(column).shape[0] for column in X[:, np.where(categorical)].T]
		if len(symbols_per_column) > 0:
			features["SymbolsMin"] = np.min(symbols_per_column)
			features["SymbolsMax"] = np.max(symbols_per_column)
			features["SymbolsMean"] = np.mean(symbols_per_column)
			features["SymbolsSTD"] = np.std(symbols_per_column)
			features["SymbolsSum"] = np.sum(symbols_per_column)
		else:
			features["SymbolsMin"] = features["SymbolsMax"] = features["SymbolsMean"] = features["SymbolsSTD"] = features["SymbolsSum"] = 0

	features["SimpleFeatureTime"] = sw.duration
	# Missing value features missing for now since only datasets without missing features were selected.

	return features

def statistical_metafeature_names():
	return ["KurtosisMin", "KurtosisMax", "KurtosisMean", "KurtosisSTD", "KurtosisKurtosis", "KurtosisSkewness",
			"SkewnessMin", "SkewnessMax", "SkewnessMean", "SkewnessSTD", "SkewnessKurtosis", "SkewnessSkewness", 
			"MeanSTDOfNumerical", "STDSTDOfNumerical", "StatisticalFeatureTime"]

def statistical_metafeatures(X, y, categorical):
	utils.input_check(X, y, categorical)
	features = OrderedDict()

	numerical = [not cat for cat in categorical]

	# Statistical meta-features are only for the numerical attributes, if there are none, we list them as -1
	# we should see if there is a better way to deal with this, as -1 is a valid value for some of these features..
	if(sum(numerical) == 0):
		return OrderedDict.fromkeys(statistical_metafeature_names(), value = -1)

	with utils.stopwatch() as sw:
		# Taking taking kurtosis of kurtosis and skewness of kurtosis is suggested by Reif et al. in Meta2-features (2012)
		kurtosisses = [scipy.stats.kurtosis(column[0]) for column in X[:,np.where(numerical)].T]	
		features["KurtosisMin"] = np.min(kurtosisses)
		features["KurtosisMax"] = np.max(kurtosisses)
		features["KurtosisMean"] = np.mean(kurtosisses)
		features["KurtosisSTD"] = np.std(kurtosisses)
		features["KurtosisKurtosis"] = scipy.stats.kurtosis(kurtosisses)
		features["KurtosisSkewness"] = scipy.stats.skew(kurtosisses)

		skewnesses = [scipy.stats.skew(column[0]) for column in X[:,np.where(numerical)].T]
		features["SkewnessMin"] = np.min(skewnesses)
		features["SkewnessMax"] = np.max(skewnesses)
		features["SkewnessMean"] = np.mean(skewnesses)
		features["SkewnessSTD"] = np.std(skewnesses)
		features["SkewnessKurtosis"] = scipy.stats.kurtosis(skewnesses)
		features["SkewnessSkewness"] = scipy.stats.skew(skewnesses)

		standard_deviations = [np.std(column[0]) for column in X[:,np.where(numerical)].T]
		features["MeanSTDOfNumerical"] = np.mean(standard_deviations)
		features["STDSTDOfNumerical"] = np.std(standard_deviations)

	features["StatisticalFeatureTime"] = sw.duration	

	return features

def information_theoretic_metafeature_names():
	return ["ClassEntropy", "MeanFeatureEntropy", "MeanMutualInformation", "NoiseToSignalRatio", "InformationFeatureTime"]

def information_theoretic_metafeatures(X, y, categorical):
	utils.input_check(X, y, categorical)
	features = OrderedDict()

	classes, counts = np.unique(y, return_counts = True)
	features["ClassEntropy"] = scipy.stats.entropy(counts, base = 2)

	# Information theoretic meta-features below only apply to categorical values
	if(sum(categorical) == 0):
		return OrderedDict.fromkeys(information_theoretic_metafeature_names(), value = -1)

	with utils.stopwatch() as sw:
		feature_entropies = [scipy.stats.entropy(column[0]) for column in X[:,np.where(categorical)].T]
		mean_feature_entropy = np.mean(feature_entropies)
		features["MeanFeatureEntropy"] = np.mean(mean_feature_entropy)

		mutual_informations = [sklearn.metrics.mutual_info_score(y, column[0]) for column in X[:, np.where(categorical)].T]
		mean_mutual_information = np.mean(mutual_informations)
		features["MeanMutualInformation"] = mean_mutual_information

		if(mean_mutual_information == 0):
			features["NoiseToSignalRatio"] = 0

		features["NoiseToSignalRatio"] = (mean_feature_entropy - mean_mutual_information) / mean_mutual_information

	features["InformationFeatureTime"] = sw.duration
	return features

if __name__ == "__main__":
	utils.log("Running tests - Importing...")
	from openml import datasets, tasks

	# Take 59 is for dataset 61, the iris dataset, which is good for numerical tests,
	# Task 60 is for dataset 62, a zoo dataset, which contains a lot of categorical information.
	task = tasks.get_task(60)
	data = task.get_dataset()
	X, y, categorical = data.get_data(target = data.default_target_attribute, return_categorical_indicator = True)

	# We want to do cross-validation for some landmarkers, so we take a cv-10 fold.
	# We need to unroll the generator into a list because it is iterated over multiple times.
	folds = list(next(task.iterate_repeats()))

	simple = simple_metafeatures(X, y, categorical)
	stats = statistical_metafeatures(X, y, categorical)
	info = information_theoretic_metafeatures(X, y, categorical)
	landmarkers = landmarker_metafeatures(X, y, categorical, folds)

	for key, val in simple.items():
		print("{}: {}".format(key, val))

	for key, val in stats.items():
		print("{}: {}".format(key, val))

	for key, val in info.items():
		print("{}: {}".format(key, val))

	for key, val in landmarkers.items():
		print("{}: {}".format(key, val))

	print("Total of {} metafeatres".format(len(simple)+len(stats)+len(info)+len(landmarkers)))