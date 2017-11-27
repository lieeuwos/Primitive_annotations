import time
import traceback
import numpy as np

def input_check(X, y, categorical):	
	assert X.shape[1] == len(categorical), "There should be as many features as categorical indicators"
	assert X.shape[0] == y.shape[0], "There should be equally many instances in X and y"
	return #Everything seems fine, carry on

def log_traceback(file_name):	
	with open(file_name, "a") as fh:
		traceback.print_exc(file=fh)

def log(message, file_name = None, level = 0,):
	if(file_name is not None):
		with open(file_name, "a") as fh:
			fh.write(message)

	print(message)

def array_except(arr, rows_to_exclude = [], columns_to_exclude = []):
	new_arr = np.delete(arr, rows_to_exclude, axis=0)
	if len(arr.shape) == 2:
		new_arr = np.delete(new_arr, columns_to_exclude, axis=1)
	return new_arr

def load_file(filename, rows_to_ignore = [], columns_to_ignore=[], seperator=","):
	with open(filename, "r") as fh:
		X = [line.split(seperator) for line in fh.readlines()]	
	return array_except(np.array(X), rows_to_ignore, columns_to_ignore)

def root_mean_squared_error(x0, x1):
	assert x0.shape == x1.shape, "x0 and x1 should be of the same shape."

	return np.sqrt(np.mean([(x[0] - x[1])**2 for x in zip(x0,x1)]))

def mean_absolute_deviation(x0, x1):
	assert x0.shape == x1.shape, "x0 and x1 should be of the same shape."

	return np.mean([abs(x[0] - x[1]) for x in zip(x0,x1)])

def numpy_to_latex(arr, column_names, row_names):
	template = "\\begin{{table}}\n\\begin{{tabular}}{column_format} \hline \n{body}\n\\end{{tabular}}\n\\caption{{caption}}\n\\label{{label}}\n\\end{{table}}\n"
	column_format = "{{|l|{}|}}".format("|".join(['l' for column in column_names]))
	columns = " & ".join(column_names) + "\\\ \hline \n"	
	body = "".join(["{} & {} \\\ \hline\n".format(row_names[i], " & ".join(row)) for i, row in enumerate(arr)])
	body = columns + body
	return template.format(column_format = column_format, body=body)


class stopwatch:
	def __enter__(self):
		self.start = time.clock()
		return self

	def __exit__(self, type, value, tb):
		self.end = time.clock()
		self.duration = self.end - self.start

	def elapsed_time(self):
		if self.end is None:
			return time.clock() - self.start
		else:
			return self.duration

