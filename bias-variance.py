from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
sh = StratifiedShuffleSplit(n_splits=10, random_state=0)

# Random Forest
rf = RandomForestClassifier(random_state=0, n_estimators=512)
scores = cross_val_score(rf, X, y, scoring="roc_auc", cv=sh)
print("Random forest AUC: %.3f" %(scores.mean()))

# Decision tree (defaults)
tree = DecisionTreeClassifier(random_state=0)
scores = cross_val_score(tree, X, y, scoring="roc_auc", cv=sh)
print("Decision tree AUC: {:.3f}".format(scores.mean()))

from sklearn.model_selection import ShuffleSplit

# Bias-Variance Computation
def compute_bias_variance(clf, X, y):
    # Bootstraps
    n_repeat = 40
    shuffle_split = ShuffleSplit(test_size=0.33, n_splits=n_repeat)

    # Store sample predictions
    y_all_pred = [[] for _ in range(len(y))]

    # Train classifier on each bootstrap and score predictions
    for i, (train_index, test_index) in enumerate(shuffle_split.split(X)):
        # Train and predict
        clf.fit(X[train_index], y[train_index])
        y_pred = clf.predict(X[test_index])

        # Store predictions
        for j,index in enumerate(test_index):
            y_all_pred[index].append(y_pred[j])

    # Compute bias, variance, error
    bias_sq = sum([ (1 - x.count(y[i])/len(x))**2 * len(x)/n_repeat
                for i,x in enumerate(y_all_pred)])
    var = sum([((1 - ((x.count(0)/len(x))**2 + (x.count(1)/len(x))**2))/2) * len(x)/n_repeat
               for i,x in enumerate(y_all_pred)])

    return np.sqrt(bias_sq), var

# Random Forest
print("Random forest Bias: %.3f, Variance: %.3f" % (compute_bias_variance(rf, X, y)))

# Decision tree (defaults)
print("Decision tree Bias: %.3f, Variance: %.3f" % (compute_bias_variance(tree, X, y)))

bias_scores = []
var_scores = []
n_estimators= [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

for i in n_estimators:
    b,v = compute_bias_variance(RandomForestClassifier(random_state=0,n_estimators=i,n_jobs=-1),X,y)
    bias_scores.append(b)
    var_scores.append(v)

plt.figure(figsize=(5,2))
plt.plot(n_estimators, var_scores,label ="variance" )
plt.plot(n_estimators, np.square(bias_scores),label ="bias^2")
plt.xscale('log',basex=2)
plt.xlabel("n_estimators")
plt.legend(loc="best")
plt.show()
