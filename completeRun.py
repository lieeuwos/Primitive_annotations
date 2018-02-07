# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 09:48:47 2017

@author: s127788
"""

from Noise2 import add_noise_features2,distr_guessed,add_noise_features,remove_features
from LocalDatasets import checkForExistFile,read_did,read_did_cat,saveSingleDict,savePredictsScore,saveEstimator
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from utils import stopwatch
from Noise2 import shuffle_set,random_test_set4,random_test_set6,random_test_set7,random_test_set8,random_test_set9,random_test_set3,split,noise_set2,add_copy_features,add_identifiers,split_identifiers,add_copy,orderX,reduce_dataset,remove_features2,create_features
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from random import random
from scipy.stats import expon
#import psutil


# for all clfs do typ features
# @did is the dataset
# @cv is the amount of splits
# @amount is the feature difference
# @ typ ==0  remove features
# @ typ == 1 add symbolic features
# @ typ == 2 add features balanced by initial catagorical and symbolic features
def featureClf(did,cv,amount,typ):
    X,y = read_did(did)
    cat = read_did_cat(did)
    if typ == 0:
        func = 'cvScoreFeatures1'
    elif typ == 1:
        func = 'cvScoreFeatures3'
    elif typ == 2:
        func = 'cvScoreFeatures4'
    elif typ == 3:
        func = 'cvScoreFeatures5'
    elif typ == 4:
        func = 'cvScoreFeatures6'
    elif typ == 5:
        func = 'removedFeatures'
    elif typ == 6:
        func = 'OnlyNoisyFeatures' 
#    func = 'TestcvScoreFeatures4'
    clfNames = ['RandomForestClassifier','KNeighborsClassifier', '1NeighborsClassifier', 'SGDClassifier', 'AdaBoost', 'SVC-rbf','GaussianNB', 'BernoulliNB','GradientBoost']
#    clfNames = ['GradientBoost']
    clf = []
    scorings = []
    score = []
    predict = []
    guessed = []
    predicts = []
    time = []
    for clfName in clfNames:
        clf.append(clfs(clfName))
        scorings.append([[],[]])
        score.append([[],[]])
        predict.append([[],[]])
        guessed.append([[],[]])
        predicts.append([[],[],[]])
        time.append([0,0,0,0])
            
    X,y = shuffle_set(X,y)
    sc = 2
    for i in range(0,cv):        
        if i == 0 or i== 9:
            if i== 0 :
                X_train = X[0:len(X)-len(X)//cv]
                X_test = X[len(X)-len(X)//cv:len(X)]
                y_train = y[0:len(y)-len(y)//cv]
                y_test = y[len(y)-len(y)//cv:len(y)]
            else:
                X_train = X[len(X)//cv:len(X)]
                X_test = X[0:len(X)//cv]
                y_train = y[len(y)//cv:len(y)]
                y_test = y[0:len(y)//cv]
        else:
            X_train = X[0:len(X)//cv*i]
            X_train.extend(X[len(X)//cv*(i+1):len(X)])
            X_test = X[len(X)//cv*i:len(X)//cv*(i+1)]
            y_train = y[0:len(y)//cv*i]
            y_train.extend(y[len(y)//cv*(i+1):len(y)])
            y_test = y[len(y)//cv*i:len(y)//cv*(i+1)]
        train_X = add_type(X_train,cat,amount,typ)
        test_X = add_type(X_test,cat,amount,typ)
        j = 0
        for clfName in clfNames:
            cv_clf = clfs(clfName)
            cv_clf2 = clfs(clfName)
            with stopwatch() as sw:
                _ = cv_clf.fit(X_train,y_train)
            time[j][0] = time[j][0] + sw.duration
            with stopwatch() as sw:
                predict[j][0] = cv_clf.predict(X_test)
            time[j][1] = time[j][1] + sw.duration
            for k in range(0,sc):
                score[j][k] = 0
            with stopwatch() as sw:    
                _ = cv_clf2.fit(train_X,y_train)
            time[j][2] = time[j][2] + sw.duration
            with stopwatch() as sw:
                predict[j][1] = cv_clf2.predict(test_X)
            time[j][3] = time[j][3] + sw.duration            
            for k in range(0,sc):
                guessed[j][k].append(distr_guessed(predict[j][k]))        
                        
            for k in range(0,sc):
                scorings[j][k].append(accuracy_score(y_test,predict[j][k]))
            for k in range(0,sc):
                predicts[j][k].append(predict[j][k])
            predicts[j][sc].append(y_test)
            j = j + 1
        
    j = 0
    for clfName in clfNames:            
        count = checkForExistFile(func,clfName,did,amount)
        if count >= 0:
            saveSingleDict(scorings[j],func,clfName,did,amount,'scores' + str(count))
            saveSingleDict(guessed[j],func,clfName,did,amount,'SummaryGuesses' + str(count))
            savePredictsScore(predicts[j],func,clfName,did,amount,'Predictions' + str(count))
            saveSingleDict([time[j]],func,clfName,did,amount,'duration' + str(count))
        j = j + 1
            
def clfs(clfName):
    if (clfName == 'RandomForestClassifier'):
        clf = RandomForestClassifier()
    elif (clfName == 'KNeighborsClassifier'):
        clf = KNeighborsClassifier()
    elif (clfName == '1NeighborsClassifier'):
        clf =  KNeighborsClassifier(n_neighbors=1)  
    elif (clfName == 'SGDClassifier'):
        clf = SGDClassifier()
    elif (clfName == 'AdaBoost'):
        clf = AdaBoostClassifier()
    elif (clfName[:4] == 'SVC-'):
        clf = SVC(kernel = clfName[4:])
        if (clfName == 'SVC-'):
            clf = SVC(kernel = 'rbf')
    elif (clfName == 'GaussianNB'):
        clf = GaussianNB()
    elif (clfName == 'BernoulliNB'):
        clf = BernoulliNB()
    elif (clfName == 'MultinomialNB'):
        clf = MultinomialNB()
    elif (clfName == 'GradientBoost'):
        clf = GradientBoostingClassifier()
    return clf

def add_type(X,cat,amount,typ):
    if typ == 0:
        return remove_features(X,amount)
    elif typ == 1:
        return add_noise_features(X,amount)
    elif typ == 2:
        return add_noise_features2(X,cat,amount)
    elif typ == 3:
        return add_copy_features(X,amount)
    elif typ == 4:
        return add_copy(X,amount)
    elif typ == 5:
        return remove_features2(X,amount)
    elif typ == 6:
        X2 = []
        for i in range(len(X)):
            X2.append([])
        return add_noise_features2(X2,cat,amount)
    
def cv_scores_noise(did,cv,amount,cvScore):
    X,y = read_did(did)
    catagorical = read_did_cat(did)
    X,y = shuffle_set(X,y) 
    clfNames = ['RandomForestClassifier','KNeighborsClassifier', '1NeighborsClassifier', 'SGDClassifier', 'AdaBoost', 'SVC-rbf','GradientBoost']#, 'SVC-linear','SVC-poly']#, 'GaussianNB', 'BernoulliNB', 'MultinomialNB']
    scorings = []
    score = []
    predict = []
    guessed = []
    predicts = []
    sc = 4
    for j in clfNames:
        scorings.append([[],[],[],[]])
        score.append([[],[],[],[]])
        predict.append([[],[],[],[]])
        guessed.append([[],[],[],[]])
        predicts.append([[],[],[],[],[]])    
    for i in range(0,cv):
        j = 0
#        cv_clf = copy(clf)
#        cv_clf2 = copy(clf)
        X_train,y_train,X_test,y_test = cv_noise_splits(X,y,i,cv)
        train_X,test_X = make_noise(X_train,X_test,y,catagorical,amount,cvScore) 
        for j in range(0,len(clfNames)):
            cv_clf = clfs(clfNames[j])
            _ = cv_clf.fit(X_train,y_train)
            predict[j][0] = cv_clf.predict(X_test)
            predict[j][1] = cv_clf.predict(test_X)
            cv_clf = clfs(clfNames[j])
            _ = cv_clf.fit(train_X,y_train)
            predict[j][2] = cv_clf.predict(test_X)
            predict[j][3] = cv_clf.predict(X_test) 
                       
            for k in range(0,sc):
                guessed[j][k].append(distr_guessed(predict[j][k]))
            for k in range(0,sc):
                predicts[j][k].append(predict[j][k])
                scorings[j][k].append(accuracy_score(y_test,predict[j][k]))
            predicts[j][sc].append(y_test)
            j = j + 1
    j = 0
    func = 'cvScoreNoise' + str(cvScore)
    for clfName in clfNames:                    
        saveSingleDict(scorings[j],func,clfName,amount,did,'scores')
        saveSingleDict(guessed[j],func,clfName,amount,did,'SummaryGuesses')
        savePredictsScore(predicts[j],func,clfName,amount,did,'Predictions')
        j = j + 1  
        
        
    
    
    
def make_noise(X_train,X_test,y,catagorical,amount,cvScore):
    if cvScore == 2:
        if amount == 0:                
            train_X = random_test_set4(X_train,y)
            test_X = random_test_set4(X_test,y)
        else:
            train_X = random_test_set3(X_train,amount)
            test_X = random_test_set3(X_test,amount)
    elif cvScore == 3:
        if amount == 0:
            train_X = random_test_set4(X_train,y)
            test_X = random_test_set4(X_test,y)
        else:
            train_X = random_test_set6(X_train,catagorical,amount)
            test_X = random_test_set6(X_test,catagorical,amount)
    elif cvScore == 4:
        if amount == 0:                
            train_X = random_test_set4(X_train,y)
            test_X = random_test_set4(X_test,y)
        else:
            train_X = random_test_set7(X_train,amount)
            test_X = random_test_set7(X_test,amount)
    elif cvScore == 5:
        if amount == 0:                
            train_X = random_test_set9(X_train,catagorical,amount)
            test_X = random_test_set9(X_test,catagorical,amount)
        else:
            train_X = random_test_set8(X_train,catagorical,amount)
            test_X = random_test_set8(X_test,catagorical,amount)
    return train_X,test_X

def cv_noise_splits(X,y,i,cv):
    if i == 0 or i== 9:
        if i== 0 :
            X_train = X[0:len(X)-len(X)//cv]
            X_test = X[len(X)-len(X)//cv:len(X)]
            y_train = y[0:len(y)-len(y)//cv]
            y_test = y[len(y)-len(y)//cv:len(y)]
        else:
            X_train = X[len(X)//cv:len(X)]
            X_test = X[0:len(X)//cv]
            y_train = y[len(y)//cv:len(y)]
            y_test = y[0:len(y)//cv]
    else:#if i=== 0 or i== 9
        X_train = X[0:len(X)//cv*i]
        X_train.extend(X[len(X)//cv*(i+1):len(X)])
        X_test = X[len(X)//cv*i:len(X)//cv*(i+1)]
        y_train = y[0:len(y)//cv*i]
        y_train.extend(y[len(y)//cv*(i+1):len(y)])
        y_test = y[len(y)//cv*i:len(y)//cv*(i+1)]            
        
    return X_train,y_train,X_test,y_test


    
def cv_scores_BrainWebb(did,cv,amount,typ):
    X,y = read_did(did)
    func = 'cvScoresScales'
#    func = 'TestcvScoreFeatures4'
    clfNames = ['RandomForestClassifier','KNeighborsClassifier', '1NeighborsClassifier', 'SGDClassifier', 'AdaBoost', 'SVC-rbf', 'GaussianNB', 'BernoulliNB']
    clf = []
    scorings = []
    score = []
    predict = []
    guessed = []
    predicts = []
    time = []
    for clfName in clfNames:
        clf.append(clfs(clfName))
        scorings.append([[]])
        score.append([[]])
        predict.append([[]])
        guessed.append([[]])
        predicts.append([[],[]])
        time.append([0,0])
            
    X,y = shuffle_set(X,y)
    X,y  = split(X,y,amount)
    sc = 1
    for i in range(0,cv):        
        X_train,y_train,X_test,y_test = cv_noise_splits(X,y,i,cv)
        j = 0
        for clfName in clfNames:
            cv_clf = clfs(clfName)
            with stopwatch() as sw:
                _ = cv_clf.fit(X_train,y_train)
            time[j][0] = time[j][0] + sw.duration
            with stopwatch() as sw:
                predict[j][0] = cv_clf.predict(X_test)
            time[j][1] = time[j][1] + sw.duration
            for k in range(0,sc):
                score[j][k] = 0                        
            for k in range(0,sc):
                guessed[j][k].append(distr_guessed(predict[j][k]))      
                        
            for k in range(0,sc):
                scorings[j][k].append(accuracy_score(y_test,predict[j][k]))
            for k in range(0,sc):
                predicts[j][k].append(predict[j][k])
            predicts[j][sc].append(y_test)
            j = j + 1
        
    j = 0
    for clfName in clfNames:            
        count = checkForExistFile(func,clfName,did,amount)
        if count >= 0:
            saveSingleDict(scorings[j],func,clfName,did,amount,'scores' + str(count))
            saveSingleDict(guessed[j],func,clfName,did,amount,'SummaryGuesses' + str(count))
            savePredictsScore(predicts[j],func,clfName,did,amount,'Predictions' + str(count))
            saveSingleDict([time[j]],func,clfName,did,amount,'duration' + str(count))
        j = j + 1


def cv_feature(did,cv,amount):
    X,y = read_did(did)
    cat = read_did_cat(did)
    func = 'cvFeatureCAT2'
#    func = 'TestcvScoreFeatures4'
    clfNames = ['GradientBoost']
#    'RandomForestClassifier','KNeighborsClassifier', '1NeighborsClassifier', 'SGDClassifier', 'AdaBoost', 'SVC-rbf', 'GaussianNB', 'BernoulliNB'
    clf = []
    scorings = []
    score = []
    predict = []
    guessed = []
    predicts = []
    time = []
    for clfName in clfNames:
        clf.append(clfs(clfName))
        scorings.append([[],[]])
        score.append([[],[]])
        predict.append([[],[]])
        guessed.append([[],[]])
        predicts.append([[],[],[]])
        time.append([0,0,0])
            
    X,y = shuffle_set(X,y)
    sc = 2
    for i in range(0,cv):        
        X_train,y_train,X_test,y_test = cv_noise_splits(X,y,i,cv)
        test_X = noise_set2(X_test,cat,amount)
        j = 0
        for clfName in clfNames:
            cv_clf = clfs(clfName)
            with stopwatch() as sw:
                _ = cv_clf.fit(X_train,y_train)
            time[j][0] = time[j][0] + sw.duration
            with stopwatch() as sw:
                predict[j][0] = cv_clf.predict(X_test)
            time[j][1] = time[j][1] + sw.duration
            with stopwatch() as sw:
                predict[j][1] = cv_clf.predict(test_X)
            time[j][2] = time[j][2] + sw.duration
            for k in range(0,sc):
                score[j][k] = 0                        
            for k in range(0,sc):
                guessed[j][k].append(distr_guessed(predict[j][k]))      
                        
            for k in range(0,sc):
                scorings[j][k].append(accuracy_score(y_test,predict[j][k]))
            for k in range(0,sc):
                predicts[j][k].append(predict[j][k])
            predicts[j][sc].append(y_test)
            j = j + 1
        
    j = 0
    for clfName in clfNames:            
        count = checkForExistFile(func,clfName,did,amount)
        if count >= 0:
            saveSingleDict(scorings[j],func,clfName,did,amount,'scores' + str(count))
            saveSingleDict(guessed[j],func,clfName,did,amount,'SummaryGuesses' + str(count))
            savePredictsScore(predicts[j],func,clfName,did,amount,'Predictions' + str(count))
            saveSingleDict([time[j]],func,clfName,did,amount,'duration' + str(count))
        j = j + 1
        
def optimizeCVGBC(did,amount,cv):
    X,y = read_did(did)
    cat = read_did_cat(did)
    sc = 2
    iters = 40
    func = 'cvOptimizeGBC'
    time = [0,0,0]
    estimator = []
    predicts = [[],[],[]]
    scorings = [[],[]]
    clfName = 'GradientBoost'
    X,y = shuffle_set(X,y)
    for i in range(0,cv):        
        X_train,y_train,X_test,y_test = cv_noise_splits(X,y,i,cv)
        test_X = noise_set2(X_test,cat,amount)
        clf = GradientBoostingClassifier()
        learns = [random()*1.9 + 0.1 for i in range(iters*4)]
        learns = sorted(learns)
        params = {'learning_rate': learns}
        cv_clf = RandomizedSearchCV(clf, param_distributions=params,
                                       n_iter=iters,n_jobs = 3)
        with stopwatch() as sw:
            _ = cv_clf.fit(X_train,y_train)
        time[0] = time[0] + sw.duration
        with stopwatch() as sw:
            predicts[0].append(cv_clf.predict(X_test))
        time[1] = time[1] + sw.duration
        with stopwatch() as sw:
            predicts[1].append(cv_clf.predict(test_X))
        time[2] = time[2] + sw.duration
        estimator.append(cv_clf.best_estimator_)
        predicts[sc].append(y_test)
        for k in range(0,sc):
            scorings[k].append(accuracy_score(y_test,predicts[k][i]))
    count = checkForExistFile(func,clfName,did,amount)
    if count >= 0:
        saveSingleDict(scorings,func,clfName,did,amount,'scores' + str(count))
        saveEstimator(str(estimator),func,clfName,did,amount,'Estimators' + str(count))
        savePredictsScore(predicts,func,clfName,did,amount,'Predictions' + str(count))
        saveSingleDict([time],func,clfName,did,amount,'duration' + str(count))
        
def optimizeCVclf(did,amount,cv,clfName):
    X,y = read_did(did)
    cat = read_did_cat(did)
    sc = 2
    iters = 40
    func = 'cvOptimizeSTD'
    time = [0,0,0]
    estimator = []
    predicts = [[],[],[]]
    scorings = [[],[]]
    X,y = shuffle_set(X,y)
    for i in range(0,cv):        
        X_train,y_train,X_test,y_test = cv_noise_splits(X,y,i,cv)
        test_X = noise_set2(X_test,cat,amount)
        cv_clf = optimizeCLF(clfName,len(X[0]),iters)
        with stopwatch() as sw:
            _ = cv_clf.fit(X_train,y_train)
        time[0] = time[0] + sw.duration
        with stopwatch() as sw:
            predicts[0].append(cv_clf.predict(X_test))
        time[1] = time[1] + sw.duration
        with stopwatch() as sw:
            predicts[1].append(cv_clf.predict(test_X))
        time[2] = time[2] + sw.duration
        estimator.append(cv_clf.best_estimator_)
        predicts[sc].append(y_test)
        for k in range(0,sc):
            scorings[k].append(accuracy_score(y_test,predicts[k][i]))
    count = checkForExistFile(func,clfName,did,amount)
    if count >= 0:
        saveSingleDict(scorings,func,clfName,did,amount,'scores' + str(count))
        saveEstimator(str(estimator),func,clfName,did,amount,'Estimators' + str(count))
        savePredictsScore(predicts,func,clfName,did,amount,'Predictions' + str(count))
        saveSingleDict([time],func,clfName,did,amount,'duration' + str(count))

def optimizeCVclfs(did,amount,cv):
    X,y = read_did(did)
    cat = read_did_cat(did)
    sc = 2
    iters = 40
    func = 'cvOptimizeSTD'
    clfNames = ['SVC-','GradientBoost','RandomForestClassifier', 'AdaBoost','KNeighborsClassifier']
    time = []
    estimator = []
    predicts = []
    scorings = []
    for clfName in clfNames:
        scorings.append([[],[]])
        estimator.append([])
        predicts.append([[],[],[]])
        time.append([0,0,0])    
    X,y = shuffle_set(X,y)
    for i in range(0,cv):        
        X_train,y_train,X_test,y_test = cv_noise_splits(X,y,i,cv)
        test_X = noise_set2(X_test,cat,amount)
        for j,clfName in enumerate(clfNames):
            cv_clf = optimizeCLF(clfName,len(X[0]),iters)
            with stopwatch() as sw:
                _ = cv_clf.fit(X_train,y_train)
            time[j][0] = time[j][0] + sw.duration
            with stopwatch() as sw:
                predicts[j][0].append(cv_clf.predict(X_test))
            time[j][1] = time[j][1] + sw.duration
            with stopwatch() as sw:
                predicts[j][1].append(cv_clf.predict(test_X))
            time[j][2] = time[j][2] + sw.duration
            estimator[j].append(cv_clf.best_estimator_)
            predicts[j][sc].append(y_test)
            for k in range(0,sc):
                scorings[j][k].append(accuracy_score(y_test,predicts[j][k][i]))
    for j,clfName in enumerate(clfNames):
        count = checkForExistFile(func,clfName,did,amount)
        if count >= 0:
            saveSingleDict(scorings[j],func,clfName,did,amount,'scores' + str(count))
            saveEstimator(str(estimator[j]),func,clfName,did,amount,'Estimators' + str(count))
            savePredictsScore(predicts[j],func,clfName,did,amount,'Predictions' + str(count))
            saveSingleDict([time[j]],func,clfName,did,amount,'duration' + str(count))




      
def optimizeCLF(clfName,maxFeatures,iters):
    if (clfName == 'RandomForestClassifier'):
        clf = RandomForestClassifier()
        params = {'max_features': range(1,maxFeatures), 'min_samples_split': range(2,20)}
        cv_clf = RandomizedSearchCV(clf, param_distributions=params,
                                           n_iter=iters,n_jobs = -1)
    elif (clfName == 'KNeighborsClassifier'):
        clf = KNeighborsClassifier()
        params = {'weights': ['uniform','distance'], 'n_neighbors' : range(1,50), 'p' : [1,2]}
        cv_clf = RandomizedSearchCV(clf, param_distributions=params,
                                       n_iter=iters,n_jobs = -1)
    elif (clfName == '1NeighborsClassifier'):
        clf =  KNeighborsClassifier(n_neighbors=1)  
    elif (clfName == 'SGDClassifier'):
        clf = SGDClassifier()
    elif (clfName == 'AdaBoost'):
        clf = AdaBoostClassifier()
        learns = [random()*1.9 + 0.1 for i in range(iters*4)]
        params = {'learning_rate': learns}
        cv_clf = RandomizedSearchCV(clf, param_distributions=params,
                                           n_iter=iters,n_jobs = -1)
    elif (clfName[:4] == 'SVC-'):
        param_grid_rbf = {'C': expon(scale=100), 
              'gamma': expon(scale=.1), 'kernel' : ['rbf', 'sigmoid']}
        cv_clf = RandomizedSearchCV(SVC(), param_distributions=param_grid_rbf,
                                           n_iter=40,n_jobs = -1)
    elif (clfName == 'GaussianNB'):
        clf = GaussianNB()
    elif (clfName == 'BernoulliNB'):
        clf = BernoulliNB()
    elif (clfName == 'GradientBoost'):
        clf = GradientBoostingClassifier()
        learns = [random()*1.9 + 0.1 for i in range(iters*4)]
        learns = sorted(learns)
        params = {'learning_rate': learns}
        cv_clf = RandomizedSearchCV(clf, param_distributions=params,
                                       n_iter=iters,n_jobs = -1)
    return cv_clf

def featureOptClf(did,cv,amount,typ):
    X,y = read_did(did)
    cat = read_did_cat(did)
    if typ == 0:
        func = 'cvOptScoreFeatures1'
    elif typ == 1:
        func = 'cvOptScoreFeatures3'
    elif typ == 2:
        func = 'cvOptScoreFeatures4'
    elif typ == 3:
        func = 'cvOptScoreFeatures5'
    elif typ == 4:
        func = 'cvOptScoreFeatures6'
    elif typ == 5:
        func = 'FeaturesOptRemoved'
#    func = 'TestcvScoreFeatures4'
    clfNames = ['SVC-','GradientBoost','RandomForestClassifier', 'AdaBoost','KNeighborsClassifier']
    iters = 40
    clf = []
    scorings = []
    score = []
    predict = []
    estimator = []
    predicts = []
    time = []
    for clfName in clfNames:
        clf.append(clfs(clfName))
        scorings.append([[],[]])
        score.append([[],[]])
        predict.append([[],[]])
        predicts.append([[],[],[]])
        time.append([0,0,0,0])
        estimator.append([])
            
    X,y = shuffle_set(X,y)
    sc = 2
    for i in range(0,cv):        
        if i == 0 or i== 9:
            if i== 0 :
                X_train = X[0:len(X)-len(X)//cv]
                X_test = X[len(X)-len(X)//cv:len(X)]
                y_train = y[0:len(y)-len(y)//cv]
                y_test = y[len(y)-len(y)//cv:len(y)]
            else:
                X_train = X[len(X)//cv:len(X)]
                X_test = X[0:len(X)//cv]
                y_train = y[len(y)//cv:len(y)]
                y_test = y[0:len(y)//cv]
        else:
            X_train = X[0:len(X)//cv*i]
            X_train.extend(X[len(X)//cv*(i+1):len(X)])
            X_test = X[len(X)//cv*i:len(X)//cv*(i+1)]
            y_train = y[0:len(y)//cv*i]
            y_train.extend(y[len(y)//cv*(i+1):len(y)])
            y_test = y[len(y)//cv*i:len(y)//cv*(i+1)]
        train_X = add_type(X_train,cat,amount,typ)
        test_X = add_type(X_test,cat,amount,typ)
        j = 0
        for clfName in clfNames:
            cv_clf = optimizeCLF(clfName,len(X_train[0]),iters)
            cv_clf2 = optimizeCLF(clfName,len(train_X[0]),iters)
            with stopwatch() as sw:
                _ = cv_clf.fit(X_train,y_train)
            time[j][0] = time[j][0] + sw.duration
            with stopwatch() as sw:
                predict[j][0] = cv_clf.predict(X_test)
            time[j][1] = time[j][1] + sw.duration
            for k in range(0,sc):
                score[j][k] = 0
            with stopwatch() as sw:    
                _ = cv_clf2.fit(train_X,y_train)
            time[j][2] = time[j][2] + sw.duration
            with stopwatch() as sw:
                predict[j][1] = cv_clf2.predict(test_X)
            time[j][3] = time[j][3] + sw.duration                  
                        
            for k in range(0,sc):
                scorings[j][k].append(accuracy_score(y_test,predict[j][k]))
            for k in range(0,sc):
                predicts[j][k].append(predict[j][k])
            predicts[j][sc].append(y_test)
            estimator[j].append(cv_clf.best_estimator_)
            estimator[j].append(cv_clf2.best_estimator_)
            j = j + 1
        
    j = 0
    for clfName in clfNames:            
        count = checkForExistFile(func,clfName,did,amount)
        if count >= 0:
            saveSingleDict(scorings[j],func,clfName,did,amount,'scores' + str(count))
            saveEstimator(str(estimator[j]),func,clfName,did,amount,'Estimators' + str(count))
            savePredictsScore(predicts[j],func,clfName,did,amount,'Predictions' + str(count))
            saveSingleDict([time[j]],func,clfName,did,amount,'duration' + str(count))
        j = j + 1
        
def featureOptClf2(did,cv,amount,typ):
    X,y = read_did(did)
    cat = read_did_cat(did)
    if typ == 0:
        func = 'cvOpt2ScoreFeatures1'
    elif typ == 1:
        func = 'cvOpt2ScoreFeatures3'
    elif typ == 2:
        func = 'cvOpt2ScoreFeatures4'
    elif typ == 3:
        func = 'cvOpt2ScoreFeatures5'
    elif typ == 4:
        func = 'cvOpt2ScoreFeatures6'
    elif typ == 5:
        func = 'FeaturesOptRemoved2'
#    func = 'TestcvScoreFeatures4'
    clfNames = ['SVC-','GradientBoost','RandomForestClassifier', 'AdaBoost','KNeighborsClassifier']
    iters = 40
    clf = []
    scorings = []
    score = []
    predict = []
    estimator = []
    predicts = []
    time = []
    for clfName in clfNames:
        clf.append(clfs(clfName))
        scorings.append([[],[]])
        score.append([[],[]])
        predict.append([[],[]])
        predicts.append([[],[],[]])
        time.append([0,0,0,0])
        estimator.append([])
    X = add_identifiers(X)        
    X,y = shuffle_set(X,y)
    X,iden = split_identifiers(X)
    sc = 2
    for i in range(0,cv):        
        if i == 0 or i== 9:
            if i== 0 :
                X_train = X[0:len(X)-len(X)//cv]
                X_test = X[len(X)-len(X)//cv:len(X)]
                y_train = y[0:len(y)-len(y)//cv]
                y_test = y[len(y)-len(y)//cv:len(y)]
            else:
                X_train = X[len(X)//cv:len(X)]
                X_test = X[0:len(X)//cv]
                y_train = y[len(y)//cv:len(y)]
                y_test = y[0:len(y)//cv]
        else:
            X_train = X[0:len(X)//cv*i]
            X_train.extend(X[len(X)//cv*(i+1):len(X)])
            X_test = X[len(X)//cv*i:len(X)//cv*(i+1)]
            y_train = y[0:len(y)//cv*i]
            y_train.extend(y[len(y)//cv*(i+1):len(y)])
            y_test = y[len(y)//cv*i:len(y)//cv*(i+1)]
        train_X = add_type(X_train,cat,amount,typ)
        test_X = add_type(X_test,cat,amount,typ)
        j = 0
        for clfName in clfNames:
            cv_clf = optimizeCLF(clfName,len(X_train[0]),iters)
            cv_clf2 = optimizeCLF(clfName,len(train_X[0]),iters)
            with stopwatch() as sw:
                _ = cv_clf.fit(X_train,y_train)
            time[j][0] = time[j][0] + sw.duration
            with stopwatch() as sw:
                predict[j][0] = cv_clf.predict(X_test)
            time[j][1] = time[j][1] + sw.duration
            for k in range(0,sc):
                score[j][k] = 0
            with stopwatch() as sw:    
                _ = cv_clf2.fit(train_X,y_train)
            time[j][2] = time[j][2] + sw.duration
            with stopwatch() as sw:
                predict[j][1] = cv_clf2.predict(test_X)
            time[j][3] = time[j][3] + sw.duration                  
                        
            for k in range(0,sc):
                scorings[j][k].append(accuracy_score(y_test,predict[j][k]))
            for k in range(0,sc):
                predicts[j][k].append(predict[j][k])
            predicts[j][sc].append(y_test)
            estimator[j].append(cv_clf.best_estimator_)
            estimator[j].append(cv_clf2.best_estimator_)
            j = j + 1
        
    j = 0
    for clfName in clfNames:            
        count = checkForExistFile(func,clfName,did,amount)
        if count >= 0:
            saveSingleDict(scorings[j],func,clfName,did,amount,'scores' + str(count))
            saveEstimator(str(estimator[j]),func,clfName,did,amount,'Estimators' + str(count))
            savePredictsScore(predicts[j],func,clfName,did,amount,'Predictions' + str(count))
            saveSingleDict([time[j]],func,clfName,did,amount,'duration' + str(count))
            saveSingleDict([iden],func,clfName,did,amount,'order' + str(count))
        j = j + 1
        
def featureOptClf3(did,cv,amount,typ):
    X,y = read_did(did)
    cat = read_did_cat(did)
    if typ == 0:
        func = 'cvOpt3ScoreFeatures1'
    elif typ == 1:
        func = 'cvOpt3ScoreFeatures3'
    elif typ == 2:
        func = 'cvOpt3ScoreFeatures4'
    elif typ == 3:
        func = 'cvOpt3ScoreFeatures5'
    elif typ == 4:
        func = 'cvOpt3ScoreFeatures6'
    elif typ == 5:
        func = 'FeaturesOptRemoved3'
#    func = 'TestcvScoreFeatures4'
    clfNames = ['SVC-','GradientBoost','RandomForestClassifier', 'AdaBoost','KNeighborsClassifier']
    iters = 40
    clf = []
    scorings = []
    score = []
    predict = []
    estimator = []
    predicts = []
    time = []
    for clfName in clfNames:
        clf.append(clfs(clfName))
        scorings.append([[],[],[]])
        score.append([[],[],[]])
        predict.append([[],[],[]])
        predicts.append([[],[],[],[]])
        time.append([0,0,0,0,0,0])
        estimator.append([])
    X = add_identifiers(X)        
    X,y = shuffle_set(X,y)
    X,iden = split_identifiers(X)
    sc = 3
    for i in range(0,cv):        
        if i == 0 or i== 9:
            if i== 0 :
                X_train = X[0:len(X)-len(X)//cv]
                X_test = X[len(X)-len(X)//cv:len(X)]
                y_train = y[0:len(y)-len(y)//cv]
                y_test = y[len(y)-len(y)//cv:len(y)]
            else:
                X_train = X[len(X)//cv:len(X)]
                X_test = X[0:len(X)//cv]
                y_train = y[len(y)//cv:len(y)]
                y_test = y[0:len(y)//cv]
        else:
            X_train = X[0:len(X)//cv*i]
            X_train.extend(X[len(X)//cv*(i+1):len(X)])
            X_test = X[len(X)//cv*i:len(X)//cv*(i+1)]
            y_train = y[0:len(y)//cv*i]
            y_train.extend(y[len(y)//cv*(i+1):len(y)])
            y_test = y[len(y)//cv*i:len(y)//cv*(i+1)]
        train_X = add_type(X_train,cat,amount,typ)
        test_X = add_type(X_test,cat,amount,typ)
        j = 0
        for clfName in clfNames:
            cv_clf = optimizeCLF(clfName,len(X_train[0]),iters)
            cv_clf2 = optimizeCLF(clfName,len(train_X[0]),iters)
            cv_clf3 = clfs(clfName)
            with stopwatch() as sw:
                _ = cv_clf.fit(X_train,y_train)
            time[j][0] = time[j][0] + sw.duration
            with stopwatch() as sw:
                predict[j][0] = cv_clf.predict(X_test)
            time[j][1] = time[j][1] + sw.duration
            for k in range(0,sc):
                score[j][k] = 0
            with stopwatch() as sw:    
                _ = cv_clf2.fit(train_X,y_train)
            time[j][2] = time[j][2] + sw.duration
            with stopwatch() as sw:
                predict[j][1] = cv_clf2.predict(test_X)
            time[j][3] = time[j][3] + sw.duration
            with stopwatch() as sw:
                _ = cv_clf3.fit(train_X,y_train)
            time[j][4] = time[j][4] + sw.duration
            with stopwatch() as sw:
                predict[j][2] = cv_clf3.predict(test_X)
            time[j][5] = time[j][5] + sw.duration
            
            
            for k in range(0,sc):
                scorings[j][k].append(accuracy_score(y_test,predict[j][k]))
            for k in range(0,sc):
                predicts[j][k].append(predict[j][k])
            predicts[j][sc].append(y_test)
            estimator[j].append(cv_clf.best_estimator_)
            estimator[j].append(cv_clf2.best_estimator_)
            j = j + 1
        
    j = 0
    for clfName in clfNames:            
        count = checkForExistFile(func,clfName,did,amount)
        if count >= 0:
            saveSingleDict(scorings[j],func,clfName,did,amount,'scores' + str(count))
            saveEstimator(str(estimator[j]),func,clfName,did,amount,'Estimators' + str(count))
            savePredictsScore(predicts[j],func,clfName,did,amount,'Predictions' + str(count))
            saveSingleDict([time[j]],func,clfName,did,amount,'duration' + str(count))
            saveSingleDict([iden],func,clfName,did,amount,'order' + str(count))
        j = j + 1
        
        
def optimizeFeatCVclf(did,cv,amount,typ,clfName):
    X,y = read_did(did)
    cat = read_did_cat(did)
    sc = 2
    iters = 40
    func = 'cvOptimizeFeatures' + str(typ + 2)
    time = [0,0,0,0]
    estimator = [[],[]]
    predicts = [[],[],[]]
    scorings = [[],[]]
    X,y = shuffle_set(X,y)
    for i in range(0,cv):        
        X_train,y_train,X_test,y_test = cv_noise_splits(X,y,i,cv)
        train_X = add_type(X_train,cat,amount,typ)
        test_X = add_type(X_test,cat,amount,typ)
        cv_clf = optimizeCLF(clfName,len(X[0]),iters)
        cv_clf2 = optimizeCLF(clfName,len(X[0]),iters)
        with stopwatch() as sw:
            _ = cv_clf.fit(X_train,y_train)
        time[0] = time[0] + sw.duration
        with stopwatch() as sw:
            predicts[0].append(cv_clf.predict(X_test))
        time[1] = time[1] + sw.duration
        with stopwatch() as sw:
            _ = cv_clf2.fit(train_X,y_train)
        time[2] = time[2] + sw.duration
        with stopwatch() as sw:
            predicts[1].append(cv_clf2.predict(test_X))
        time[3] = time[3] + sw.duration       
        estimator[0].append(cv_clf.best_estimator_)
        estimator[1].append(cv_clf2.best_estimator_)
        predicts[sc].append(y_test)
        for k in range(0,sc):
            scorings[k].append(accuracy_score(y_test,predicts[k][i]))
    count = checkForExistFile(func,clfName,did,amount)
    if count >= 0:
        saveSingleDict(scorings,func,clfName,did,amount,'scores' + str(count))
        saveEstimator(str(estimator),func,clfName,did,amount,'Estimators' + str(count))
        savePredictsScore(predicts,func,clfName,did,amount,'Predictions' + str(count))
        saveSingleDict([time],func,clfName,did,amount,'duration' + str(count))
        
        
def NoiseOptClf2(did,cv,amount):
    X,y = read_did(did)
    cat = read_did_cat(did)
    func = 'noiseOptClf2'
#    func = 'TestcvScoreFeatures4'
    clfNames = ['SVC-','GradientBoost','RandomForestClassifier', 'AdaBoost','KNeighborsClassifier']
    iters = 40
    clf = []
    scorings = []
    score = []
    predict = []
    estimator = []
    predicts = []
    time = []
    for clfName in clfNames:
        clf.append(clfs(clfName))
        scorings.append([[],[],[]])
        score.append([[],[],[]])
        predict.append([[],[],[]])
        predicts.append([[],[],[],[]])
        time.append([0,0,0,0,0])
        estimator.append([])
    X = add_identifiers(X)        
    X,y = shuffle_set(X,y)
    X,iden = split_identifiers(X)
    sc = 3
    for i in range(0,cv):        
        if i == 0 or i== 9:
            if i== 0 :
                X_train = X[0:len(X)-len(X)//cv]
                X_test = X[len(X)-len(X)//cv:len(X)]
                y_train = y[0:len(y)-len(y)//cv]
                y_test = y[len(y)-len(y)//cv:len(y)]
            else:
                X_train = X[len(X)//cv:len(X)]
                X_test = X[0:len(X)//cv]
                y_train = y[len(y)//cv:len(y)]
                y_test = y[0:len(y)//cv]
        else:
            X_train = X[0:len(X)//cv*i]
            X_train.extend(X[len(X)//cv*(i+1):len(X)])
            X_test = X[len(X)//cv*i:len(X)//cv*(i+1)]
            y_train = y[0:len(y)//cv*i]
            y_train.extend(y[len(y)//cv*(i+1):len(y)])
            y_test = y[len(y)//cv*i:len(y)//cv*(i+1)]
        test_X = noise_set2(X_test,cat,amount)
        j = 0
        for clfName in clfNames:
            cv_clf = optimizeCLF(clfName,len(X_train[0]),iters)
            cv_clf2 = clfs(clfName)
            with stopwatch() as sw:
                _ = cv_clf.fit(X_train,y_train)
            time[j][0] = time[j][0] + sw.duration
            with stopwatch() as sw:
                predict[j][0] = cv_clf.predict(X_test)
            time[j][1] = time[j][1] + sw.duration
            for k in range(0,sc):
                score[j][k] = 0
            with stopwatch() as sw:
                predict[j][1] = cv_clf.predict(test_X)
            time[j][2] = time[j][2] + sw.duration                  
            with stopwatch() as sw:
                _ = cv_clf2.fit(X_train,y_train)
            time[j][3] = time[j][3] + sw.duration
            with stopwatch() as sw:
                predict[j][2] = cv_clf.predict(test_X)
            time[j][4] = time[j][4] + sw.duration 
            
            for k in range(0,sc):
                scorings[j][k].append(accuracy_score(y_test,predict[j][k]))
            for k in range(0,sc):
                predicts[j][k].append(predict[j][k])
            predicts[j][sc].append(y_test)
            estimator[j].append(cv_clf.best_estimator_)
            j = j + 1
        
    j = 0
    for clfName in clfNames:            
        count = checkForExistFile(func,clfName,did,amount)
        if count >= 0:
            saveSingleDict(scorings[j],func,clfName,did,amount,'scores' + str(count))
            saveEstimator(str(estimator[j]),func,clfName,did,amount,'Estimators' + str(count))
            savePredictsScore(predicts[j],func,clfName,did,amount,'Predictions' + str(count))
            saveSingleDict([time[j]],func,clfName,did,amount,'duration' + str(count))
            saveSingleDict([iden],func,clfName,did,amount,'order' + str(count))
        j = j + 1
        

def optimizeIdenCVclf(did,iden,amount,cv,clf,clfName):
    X,y = read_did(did)
    cat = read_did_cat(did)
    sc = 2
    func = 'cvOptimizeSTD'
    time = [0,0,0]
    estimator = []
    predicts = [[],[],[]]
    scorings = [[],[]]
    X,y = orderX(X,y,iden)
    for i in range(0,cv):        
        X_train,y_train,X_test,y_test = cv_noise_splits(X,y,i,cv)
        test_X = noise_set2(X_test,cat,amount)
        cv_clf = clf[i]
        with stopwatch() as sw:
            _ = cv_clf.fit(X_train,y_train)
        time[0] = time[0] + sw.duration
        with stopwatch() as sw:
            predicts[0].append(cv_clf.predict(X_test))
        time[1] = time[1] + sw.duration
        with stopwatch() as sw:
            predicts[1].append(cv_clf.predict(test_X))
        time[2] = time[2] + sw.duration
        estimator.append(cv_clf)
        predicts[sc].append(y_test)
        for k in range(0,sc):
            scorings[k].append(accuracy_score(y_test,predicts[k][i]))
    count = checkForExistFile(func,clfName,did,amount)
    if count >= 0:
        saveSingleDict(scorings,func,clfName,did,amount,'scores' + str(count))
        saveEstimator(str(estimator),func,clfName,did,amount,'Estimators' + str(count))
        savePredictsScore(predicts,func,clfName,did,amount,'Predictions' + str(count))
        saveSingleDict([time],func,clfName,did,amount,'duration' + str(count))
        
        
def featureRemovedClf(did,cv,amount):
    X,y = read_did(did)
    func = 'scalability'
#    func = 'TestcvScoreFeatures4'
    clfNames = ['RandomForestClassifier','KNeighborsClassifier', '1NeighborsClassifier', 'SGDClassifier', 'AdaBoost', 'SVC-rbf','GaussianNB', 'BernoulliNB','GradientBoost']
#    clfNames = ['GradientBoost']
    clf = []
    scorings = []
    score = []
    predict = []
    guessed = []
    predicts = []
    time = []
    for clfName in clfNames:
        clf.append(clfs(clfName))
        scorings.append([[],[]])
        score.append([[],[]])
        predict.append([[],[]])
        guessed.append([[],[]])
        predicts.append([[],[],[]])
        time.append([0,0])
      
    X = add_identifiers(X)
    X,y = reduce_dataset(X,y,amount)
    X,iden = split_identifiers(X)
    sc = 1
    for i in range(0,cv):        
        if i == 0 or i== 9:
            if i== 0 :
                X_train = X[0:len(X)-len(X)//cv]
                X_test = X[len(X)-len(X)//cv:len(X)]
                y_train = y[0:len(y)-len(y)//cv]
                y_test = y[len(y)-len(y)//cv:len(y)]
            else:
                X_train = X[len(X)//cv:len(X)]
                X_test = X[0:len(X)//cv]
                y_train = y[len(y)//cv:len(y)]
                y_test = y[0:len(y)//cv]
        else:
            X_train = X[0:len(X)//cv*i]
            X_train.extend(X[len(X)//cv*(i+1):len(X)])
            X_test = X[len(X)//cv*i:len(X)//cv*(i+1)]
            y_train = y[0:len(y)//cv*i]
            y_train.extend(y[len(y)//cv*(i+1):len(y)])
            y_test = y[len(y)//cv*i:len(y)//cv*(i+1)]
        j = 0
        for clfName in clfNames:
            cv_clf = clfs(clfName)
            with stopwatch() as sw:
                _ = cv_clf.fit(X_train,y_train)
            time[j][0] = time[j][0] + sw.duration
            with stopwatch() as sw:
                predict[j][0] = cv_clf.predict(X_test)
            time[j][1] = time[j][1] + sw.duration
            for k in range(0,sc):
                score[j][k] = 0                        
            for k in range(0,sc):
                guessed[j][k].append(distr_guessed(predict[j][k]))        
                        
            for k in range(0,sc):
                scorings[j][k].append(accuracy_score(y_test,predict[j][k]))
            for k in range(0,sc):
                predicts[j][k].append(predict[j][k])
            predicts[j][sc].append(y_test)
            j = j + 1
        
    j = 0
    for clfName in clfNames:            
        count = checkForExistFile(func,clfName,did,amount)
        if count >= 0:
            saveSingleDict(scorings[j],func,clfName,did,amount,'scores' + str(count))
            saveSingleDict(guessed[j],func,clfName,did,amount,'SummaryGuesses' + str(count))
            savePredictsScore(predicts[j],func,clfName,did,amount,'Predictions' + str(count))
            saveSingleDict([time[j]],func,clfName,did,amount,'duration' + str(count))
            saveSingleDict([iden],func,clfName,did,amount,'order' + str(count))
        j = j + 1
        

def useOpt(dict1,clfName):
    if clfName == 'SVC-':
        clf = SVC(C = dict1['C'],kernel = dict1['kernel'][1:len(dict1['kernel'])-1],gamma = dict1['gamma'])
    elif clfName == 'RandomForestClassifier':
        clf = RandomForestClassifier(max_features = int(dict1['max_features']),min_samples_split = int(dict1['min_samples_split']))
    elif clfName == 'AdaBoost':
        clf = AdaBoostClassifier(learning_rate = dict1['learning_rate'])
    elif clfName == 'GradientBoost':
        clf = GradientBoostingClassifier(learning_rate = dict1['learning_rate'])
    elif clfName == 'KNeighborsClassifier':
        clf = KNeighborsClassifier(weights = dict1['weights'][1:len(dict1['weights'])-1],n_neighbors = int(dict1['n_neighbors']),p = int(dict1['p']))
    return clf


def featureYClf(did,cv,amount):
    X,y = read_did(did)
    func = 'GivenTarget' 
#    func = 'TestcvScoreFeatures4'
    clfNames = ['RandomForestClassifier','KNeighborsClassifier', '1NeighborsClassifier', 'SGDClassifier', 'AdaBoost', 'SVC-rbf','GaussianNB', 'BernoulliNB','GradientBoost']
#    clfNames = ['GradientBoost']
    clf = []
    scorings = []
    score = []
    predict = []
    guessed = []
    predicts = []
    time = []
    for clfName in clfNames:
        clf.append(clfs(clfName))
        scorings.append([[],[]])
        score.append([[],[]])
        predict.append([[],[]])
        guessed.append([[],[]])
        predicts.append([[],[],[]])
        time.append([0,0,0,0])
            
    X,y = shuffle_set(X,y)
    sc = 2
    for i in range(0,cv):        
        if i == 0 or i== 9:
            if i== 0 :
                X_train = X[0:len(X)-len(X)//cv]
                X_test = X[len(X)-len(X)//cv:len(X)]
                y_train = y[0:len(y)-len(y)//cv]
                y_test = y[len(y)-len(y)//cv:len(y)]
            else:
                X_train = X[len(X)//cv:len(X)]
                X_test = X[0:len(X)//cv]
                y_train = y[len(y)//cv:len(y)]
                y_test = y[0:len(y)//cv]
        else:
            X_train = X[0:len(X)//cv*i]
            X_train.extend(X[len(X)//cv*(i+1):len(X)])
            X_test = X[len(X)//cv*i:len(X)//cv*(i+1)]
            y_train = y[0:len(y)//cv*i]
            y_train.extend(y[len(y)//cv*(i+1):len(y)])
            y_test = y[len(y)//cv*i:len(y)//cv*(i+1)]
        train_X = create_features(y_train)
        test_X = create_features(y_test)
        j = 0
        for clfName in clfNames:
            cv_clf = clfs(clfName)
            cv_clf2 = clfs(clfName)
            with stopwatch() as sw:
                _ = cv_clf.fit(X_train,y_train)
            time[j][0] = time[j][0] + sw.duration
            with stopwatch() as sw:
                predict[j][0] = cv_clf.predict(X_test)
            time[j][1] = time[j][1] + sw.duration
            for k in range(0,sc):
                score[j][k] = 0
            with stopwatch() as sw:    
                _ = cv_clf2.fit(train_X,y_train)
            time[j][2] = time[j][2] + sw.duration
            with stopwatch() as sw:
                predict[j][1] = cv_clf2.predict(test_X)
            time[j][3] = time[j][3] + sw.duration            
            for k in range(0,sc):
                guessed[j][k].append(distr_guessed(predict[j][k]))        
                        
            for k in range(0,sc):
                scorings[j][k].append(accuracy_score(y_test,predict[j][k]))
            for k in range(0,sc):
                predicts[j][k].append(predict[j][k])
            predicts[j][sc].append(y_test)
            j = j + 1
    j = 0
    for clfName in clfNames:            
        count = checkForExistFile(func,clfName,did,amount)
        if count >= 0:
            saveSingleDict(scorings[j],func,clfName,did,amount,'scores' + str(count))
            saveSingleDict(guessed[j],func,clfName,did,amount,'SummaryGuesses' + str(count))
            savePredictsScore(predicts[j],func,clfName,did,amount,'Predictions' + str(count))
            saveSingleDict([time[j]],func,clfName,did,amount,'duration' + str(count))
        j = j + 1