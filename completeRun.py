# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 09:48:47 2017

@author: s127788
"""

from Noise2 import add_noise_features2,distr_guessed,add_noise_features,remove_features
from LocalDatasets import checkForExistFile,read_did,read_did_cat,saveSingleDict,savePredictsScore
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from utils import stopwatch
from Noise2 import shuffle_set,random_test_set4,random_test_set6,random_test_set7,random_test_set8,random_test_set9,random_test_set3,split,noise_set2
from sklearn.metrics import accuracy_score
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
    elif typ ==2:
        func = 'cvScoreFeatures4'
#    func = 'TestcvScoreFeatures4'
    clfNames = ['RandomForestClassifier','KNeighborsClassifier', '1NeighborsClassifier', 'SGDClassifier', 'AdaBoost', 'SVC-rbf']#, 'SVC-linear','SVC-poly']#, 'GaussianNB', 'BernoulliNB', 'MultinomialNB']
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
            X_train = X[0:len(X)//10*i]
            X_train.extend(X[len(X)//10*(i+1):len(X)])
            X_test = X[len(X)//10*i:len(X)//10*(i+1)]
            y_train = y[0:len(y)//10*i]
            y_train.extend(y[len(y)//10*(i+1):len(y)])
            y_test = y[len(y)//10*i:len(y)//10*(i+1)]
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
    elif (clfName == 'GaussianNB'):
        clf = GaussianNB()
    elif (clfName == 'BernoulliNB'):
        clf = BernoulliNB()
    elif (clfName == 'MultinomialNB'):
        clf = MultinomialNB()
    return clf

def add_type(X,cat,amount,typ):
    if typ == 0:
        return remove_features(X,amount)
    elif typ == 1:
        return add_noise_features(X,amount)
    elif typ == 2:
        return add_noise_features2(X,cat,amount)
    
def cv_scores_noise(did,cv,amount,cvScore):
    X,y = read_did(did)
    catagorical = read_did_cat(did)
    X,y = shuffle_set(X,y) 
    clfNames = ['RandomForestClassifier','KNeighborsClassifier', '1NeighborsClassifier', 'SGDClassifier', 'AdaBoost', 'SVC-rbf']#, 'SVC-linear','SVC-poly']#, 'GaussianNB', 'BernoulliNB', 'MultinomialNB']
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
        X_train = X[0:len(X)//10*i]
        X_train.extend(X[len(X)//10*(i+1):len(X)])
        X_test = X[len(X)//10*i:len(X)//10*(i+1)]
        y_train = y[0:len(y)//10*i]
        y_train.extend(y[len(y)//10*(i+1):len(y)])
        y_test = y[len(y)//10*i:len(y)//10*(i+1)]            
        
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