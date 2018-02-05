from __future__ import division
from collections import OrderedDict
import os

from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from openml import tasks

# Functions created myself in other documents
from landmarking import landmarker_metafeatures, landmarking_metafeature_names, subsample_metafeatures, subsample_metafeature_names
from metafeatures import simple_metafeatures, statistical_metafeatures, information_theoretic_metafeatures
import metafeatures as mf
from utils import log, log_traceback, stopwatch

# Variables defined through a "configuration" file
import config

#========================================================================
# Main - Characterize datasets by meta-features and record training time
#========================================================================
if not os.path.exists(config.output_directory):
    os.makedirs(config.output_directory)

# First write out the column names to the file.
with open(config.document_name, 'a') as fh:
	simple_names = ",".join(mf.simple_metafeature_names())
	statistical_names = ",".join(mf.statistical_metafeature_names())
	information_names = ",".join(mf.information_theoretic_metafeature_names())
	landmarking_names = ",".join(landmarking_metafeature_names())
	subsample_names = ",".join(subsample_metafeature_names())
	learner_names = ",".join([baselearner.__name__ for baselearner in config.base_learners])
	log(learner_names)
	column_names = "{},{},{},{},{},{},{}\n".format("did,subsize", simple_names, statistical_names, information_names, landmarking_names, subsample_names, learner_names)
	fh.write(column_names)

# Then for each dataset (and every desired subset of it), perform landmarking,
# and record execution time.
for task_id in config.task_ids:
	if task_id in config.excluded_tasks.keys():
		continue

	log("Getting task {}".format(task_id))
	task = tasks.get_task(task_id)
	did = task.dataset_id
	log("Loading dataset {}".format(did))

	try:
		dataset = task.get_dataset()
		# Impute the values - While values would be imputed when calculating some meta-features anyway, this gives more control.
		X, y, categorical = dataset.get_data(target=dataset.default_target_attribute,return_categorical_indicator=True)

        #X, categorical = remove_zero_columns(impute_values(X, categorical), categorical)

		# Subsample landmarker need folds, the train+test set of subsample landmarkers should be 500 instances,
		# since that is the size of our smallest dataset.
		# We first create a fold for 500 stratified samples, and then again divide that selection into 10 folds.
		max_size = 500
		number_of_classes = len(np.unique(y))
		if y.shape[0] < (max_size + number_of_classes):
			subset_indices = np.arange(max_size)
		else:
			subset_split = StratifiedShuffleSplit(n_splits=1, test_size=500, random_state = 0)
			_, subset_indices = next(subset_split.split(X,y))
		mapped_folds_split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state = 0)
		mapped_folds = mapped_folds_split.split(X[subset_indices],y[subset_indices])

		subsample_folds = [(subset_indices[train],subset_indices[test]) for train, test in mapped_folds]

		# Because the subsamples are of constant size, always 500, we just calculate them once per dataset,
		# not once for every subsample of every dataset (those are stratified anyway)
		log("subsample-mf")
		subsample_features = subsample_metafeatures(X, y, categorical, subsample_folds)

		# We also take subsets of the original dataset, because it creates a bigger metadataset to learn from
		for i in np.arange(0.1, 1.01, 0.1):

			# We want a minimum size of 500, otherwise predicting runtime is not that useful anyway,
			# and it avoids some issues with train/test splits being too small and timing not being accurately measured
			if(int(i*len(y)) >= 500):
				log("Subsample.")
				if i != 1:
					sss = StratifiedShuffleSplit(n_splits=1, test_size=(1-i), random_state = 0)
					train_indices, _ = next(sss.split(X, y))
				else:
					# We can not take a shuffle split with test size of 0
					train_indices = range(len(y))

				X_s = X[train_indices]
				y_s = y[train_indices]

				# Landmarkers need folds, because their time is measured for full cross-validation
				folds_split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state = 0)
				folds = list(folds_split.split(X_s,y_s)) # Convert to list because we'll iterate over it many times

				# Calculate meta-features
				log("simple-mf")
				simple_features = simple_metafeatures(X_s, y_s, categorical)
				log("stat-mf")
				statistical_features = statistical_metafeatures(X_s, y_s, categorical)
				log("info-mf")
				info_features = information_theoretic_metafeatures(X_s, y_s, categorical)
				log("landmark-mf")
				landmark_features = landmarker_metafeatures(X_s, y_s, categorical, folds)

				# Run baseleaner experiments
				baselearner_results = OrderedDict()
				for baselearner in config.base_learners:
					log("base-learners: {}".format(baselearner.__name__))
					with stopwatch() as sw:
						baselearner().fit(X_s, y_s)

					baselearner_results[baselearner.__name__] = sw.duration # result if type(result) is float else "E"

				with open(config.document_name, 'a') as fh:
					feature_list = [[did, i], simple_features.values(), statistical_features.values(), info_features.values(),
									 landmark_features.values(), subsample_features.values(),baselearner_results.values()]
					list_as_string = ",".join([str(item) for sublist in feature_list for item in sublist])
					fh.write(list_as_string + "\n")
				del X_s, y_s

		del X, y, categorical, dataset, task

	except Exception as err:
		log_traceback(config.logfile_name)
		log("dataset {} of task {} gives error {}\n".format(did, task_id, err), file_name = config.logfile_name)

with open(config.logfile_name, 'a') as fh:
	fh.write("Completed")
