# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 11:31:33 2017

@author: S127788
"""
from copy import copy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from Noise2 import *
from preamble import *
from LocalDatasets import read_did
import numpy as np
from sklearn.model_selection import cross_val_score
from random import shuffle
from sklearn.metrics import accuracy_score
from utils import stopwatch
local = True


def functionoid(clf,did,adj):    
    return functionoid_amount(clf,did,10,adj)

def functionoid_amount(clf,did, amount,adj):
    local = True
    listOutput = []
    if (local) :
        X,y = read_did(did)
    else:
        DataSetOML = oml.datasets.get_dataset(did)
        X, y = DataSetOML.get_data(target=DataSetOML.default_target_attribute);
     
    svm1 = clf
    svm2 = copy(svm1)
    svm3 = copy(svm1)
    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y)
    noise_X_train,noise_y_train = random_test_set4(X_train,y_train,amount)
    #len(noise_X_train)
    #len(noise_y_train)
    
    noise_X_test,noise_y_test = random_test_set4(X_test,y_test,amount)
    
    _ = svm1.fit(noise_X_train, noise_y_train)
    _ = svm2.fit(X_train, y_train)
    
    y_predict_noise1 = svm1.predict(noise_X_test)
    y_predict_noise2 = svm2.predict(noise_X_test)
    y_predict = svm2.predict(X_test)
    #print("Score of no noise in training data",svm1.score(noise_X_test,noise_y_test))
    #print("Score of noise in training and test data",svm2.score(noise_X_test,noise_y_test))
    # add noise then split up
    noise_X,noise_y = random_test_set4(X,y,amount)
    X_train3, X_test3, y_train3, y_test3 = train_test_split(noise_X, noise_y, stratify = noise_y)
    _ = svm3.fit(X_train3,y_train3)
    y_predict_noise3 = svm3.predict(X_test3)
    #print("Score of adding noise before split",svm3.score(X_test3,y_test3))
    
    #calculate noise influence and errors made
    score_1 = 0
    score_2 = 0
    score_3 = 0
    noise1 = 0
    noise2 = 0
    noise3 = 0
    score = 1
    pred_wrong = set()
    pred_wrong3 = set()
    for i in range(0,len(noise_X_test)):
        if noise_X_test[i] == [0] *len(noise_X_test[i]):
            pred_wrong.add(i)
    for i in range(0,len(noise_y_test)):
        if y_predict_noise2[i] == noise_y_test[i]:
            score_2 = score_2 + 1
            if i in pred_wrong:
                noise2 = noise2+1
        if y_predict_noise1[i] == noise_y_test[i]:
            score_1 = score_1 + 1
            if i in pred_wrong:
                noise1 = noise1+1
    listOutput.append({'score' : score_1, 'noise' : noise1})
    listOutput.append({'score' : score_2, 'noise' : noise2})
    #print("amount of wrong classification first classifier ",((len(y_test)-score_1)/len(y_test)))
    #print("amount of wrong classification second classifier ",((len(y_test)-score_2)/len(y_test)))
    #print("noise classified by first ",noise1)
    #print("noise classified by seconds", noise2)
    for i in range(0,len(X_test3)):
        if X_test3[i] == [0] *len(X_test3[i]):
            pred_wrong3.add(i)
    for i in range(0,len(y_test3)):
        if y_predict_noise3[i] == y_test3[i]:
            score_3 = score_3 + 1
            if i in pred_wrong3:
                noise3 = noise3+1
    listOutput.append({'score' : score_3, 'noise' : noise3})
    listOutput.append({'test1' : len(noise_y_test), 'test2' : len(y_test3), 
                       'amountOfTargets' : len(values_target(y)), 'noise1' : len(pred_wrong),
                       'noise2' : len(pred_wrong3)})
    #print("amount of wrong classification third classifier", ((len(y_test3)-score_3)/len(y_test3)))
    #print("noise classified by third " ,noise3)
    return listOutput



def loopFunctionoid(clf,did,times,adj):
    listOutput = functionoid(clf,did,adj)
    if (local) :
        X,y = read_did(did)
    else:
        DataSetOML = oml.datasets.get_dataset(did)
        X, y = DataSetOML.get_data(target=DataSetOML.default_target_attribute);
     
    for i in range(1,times):
        svm1 = copy(clf)
        svm2 = copy(clf)
        svm3 = copy(clf)
        X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y)
        noise_X_train,noise_y_train = add_noise(X_train,y_train,adj)
        #len(noise_X_train)
        #len(noise_y_train)
        noise_X_test,noise_y_test = add_noise(X_test,y_test,adj)
        _ = svm1.fit(noise_X_train,noise_y_train)
        _ = svm2.fit(X_train,y_train)
        
        y_predict_noise1 = svm1.predict(noise_X_test)
        y_predict_noise2 = svm2.predict(noise_X_test)
        y_predict = svm2.predict(X_test)
        #print("Score of no noise in training data",svm1.score(noise_X_test,noise_y_test))
        #print("Score of noise in training and test data",svm2.score(noise_X_test,noise_y_test))
        # add noise then split up
        noise_X,noise_y = add_noise(X,y,adj)
        X_train3, X_test3, y_train3, y_test3 = train_test_split(noise_X, noise_y, stratify = noise_y)
        _ = svm3.fit(X_train3,y_train3)
        y_predict_noise3 = svm3.predict(X_test3)
        #print("Score of adding noise before split",svm3.score(X_test3,y_test3))
        
        #calculate noise influence and errors made
        score_1 = 0
        score_2 = 0
        score_3 = 0
        noise1 = 0
        noise2 = 0
        noise3 = 0
        pred_wrong = set()
        pred_wrong3 = set()
        for i in range(0,len(noise_X_test)):
            if noise_X_test[i] == [0] *len(noise_X_test[i]):
                pred_wrong.add(i)
        for i in range(0,len(noise_y_test)):
            if y_predict_noise2[i] == noise_y_test[i]:
                score_2 = score_2 + 1
                if i in pred_wrong:
                    noise2 = noise2+1
            if y_predict_noise1[i] == noise_y_test[i]:
                score_1 = score_1 + 1
                if i in pred_wrong:
                    noise1 = noise1+1
        listOutput[0]['score'] = listOutput[0]['score'] + score_1
        listOutput[1]['score'] = listOutput[1]['score'] + score_2
        listOutput[0]['noise'] = listOutput[0]['noise'] + noise1
        listOutput[1]['noise'] = listOutput[1]['noise'] + noise2
        for i in range(0,len(X_test3)):
            if X_test3[i] == [0] *len(X_test3[i]):
                pred_wrong3.add(i)
        for i in range(0,len(y_test3)):
            if y_predict_noise3[i] == y_test3[i]:
                score_3 = score_3 + 1
                if i in pred_wrong3:
                    noise3 = noise3+1
        listOutput[2]['score'] = listOutput[2]['score'] + score_3
        listOutput[2]['noise'] = listOutput[2]['noise'] + noise3
    
    listOutput[0]['score'] = listOutput[0]['score']//times
    listOutput[1]['score'] = listOutput[1]['score']//times
    listOutput[0]['noise'] = listOutput[0]['noise']//times
    listOutput[1]['noise'] = listOutput[1]['noise']//times
    listOutput[2]['score'] = listOutput[2]['score']//times
    listOutput[2]['noise'] = listOutput[2]['noise']//times
    return listOutput


def loopFunctionoid_amount(clf,did,times,adj,amount):
    listOutput = functionoid_amount(clf,did,amount,adj)
    if (local) :
        X,y = read_did(did)
    else:
        DataSetOML = oml.datasets.get_dataset(did)
        X, y = DataSetOML.get_data(target=DataSetOML.default_target_attribute);
     
    for i in range(1,times):
        svm1 = copy(clf)
        svm2 = copy(clf)
        svm3 = copy(clf)
        X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y)
        noise_X_train,noise_y_train = add_noise_amount(X_train,y_train,amount,adj)
        #len(noise_X_train)
        #len(noise_y_train)
        noise_X_test,noise_y_test = add_noise_amount(X_test,y_test,amount,adj)
        _ = svm1.fit(noise_X_train,noise_y_train)
        _ = svm2.fit(X_train,y_train)
        
        y_predict_noise1 = svm1.predict(noise_X_test)
        y_predict_noise2 = svm2.predict(noise_X_test)
        y_predict = svm2.predict(X_test)
        #print("Score of no noise in training data",svm1.score(noise_X_test,noise_y_test))
        #print("Score of noise in training and test data",svm2.score(noise_X_test,noise_y_test))
        # add noise then split up
        noise_X,noise_y = add_noise_amount(X,y,amount,adj)
        X_train3, X_test3, y_train3, y_test3 = train_test_split(noise_X, noise_y, stratify = noise_y)
        _ = svm3.fit(X_train3,y_train3)
        y_predict_noise3 = svm3.predict(X_test3)
        #print("Score of adding noise before split",svm3.score(X_test3,y_test3))
        
        #calculate noise influence and errors made
        score_1 = 0
        score_2 = 0
        score_3 = 0
        noise1 = 0
        noise2 = 0
        noise3 = 0
        pred_wrong = set()
        pred_wrong3 = set()
        for i in range(0,len(noise_X_test)):
            if noise_X_test[i] == [0] *len(noise_X_test[i]):
                pred_wrong.add(i)
        for i in range(0,len(noise_y_test)):
            if y_predict_noise2[i] == noise_y_test[i]:
                score_2 = score_2 + 1
                if i in pred_wrong:
                    noise2 = noise2+1
            if y_predict_noise1[i] == noise_y_test[i]:
                score_1 = score_1 + 1
                if i in pred_wrong:
                    noise1 = noise1+1
        listOutput[0]['score'] = listOutput[0]['score'] + score_1
        listOutput[1]['score'] = listOutput[1]['score'] + score_2
        listOutput[0]['noise'] = listOutput[0]['noise'] + noise1
        listOutput[1]['noise'] = listOutput[1]['noise'] + noise2
        for i in range(0,len(X_test3)):
            if X_test3[i] == [0] *len(X_test3[i]):
                pred_wrong3.add(i)
        for i in range(0,len(y_test3)):
            if y_predict_noise3[i] == y_test3[i]:
                score_3 = score_3 + 1
                if i in pred_wrong3:
                    noise3 = noise3+1
        listOutput[2]['score'] = listOutput[2]['score'] + score_3
        listOutput[2]['noise'] = listOutput[2]['noise'] + noise3
    
    listOutput[0]['score'] = listOutput[0]['score']//times
    listOutput[1]['score'] = listOutput[1]['score']//times
    listOutput[0]['noise'] = listOutput[0]['noise']//times
    listOutput[1]['noise'] = listOutput[1]['noise']//times
    listOutput[2]['score'] = listOutput[2]['score']//times
    listOutput[2]['noise'] = listOutput[2]['noise']//times
    return listOutput



def functionoidFeatures(clf,did,rand,amount):
    listOutput = []
    local = True
    if (local) :
        X,y = read_did(did)
    else:
        DataSetOML = oml.datasets.get_dataset(did)
        X, y = DataSetOML.get_data(target=DataSetOML.default_target_attribute);
      
    if len(X[0]) > 10:
        feature_X = add_noise_features(X,len(X[0])//amount,rand)
    else:
        feature_X = add_noise_features(X,10//amount,rand)
    clf1 = clf
    clf2 = copy(clf)
    listOutput.append(cross_val_score(clf1, X, y,cv=10,n_jobs = -1,scoring = 'f1_weighted'))
    listOutput.append(cross_val_score(clf2, feature_X, y,cv=10,n_jobs = -1, scoring = 'f1_weighted'))
    return listOutput

def cv_scores(X,y,clf,cv):
       
    scorings = []
    for i in range(0,cv):
        cv_clf = copy(clf)
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
        
        _ = cv_clf.fit(X_train,y_train)
        predict = cv_clf.predict(X_test)
        score = 0
        for i in range(0,len(y_test)):
            if y_test[i] == predict[i]:
                score = score + 1
        scorings.append(score/len(y_test))
    
    
    return scorings


def cv_scores_noise(X,y,clf,cv,amount):
    X,y = shuffle_set(X,y) 
    scorings = []
    scorings1 = []
    scorings2 = []
    scorings3 = []
    scorings4 = []
    for i in range(0,cv):
        cv_clf = copy(clf)
        cv_clf2 = copy(clf)
        if i == 0 or i== 9:
            if i== 0 :
                X_train = X[0:len(X)-len(X)//cv]
                X_test = X[len(X)-len(X)//cv:len(X)]
                y_train = y[0:len(y)-len(y)//cv]
                y_test = y[len(y)-len(y)//cv:len(y)]
                train_X = random_test_set4(X_train,y)
                test_X = random_test_set4(X_test,y)
            else:
                X_train = X[len(X)//cv:len(X)]
                X_test = X[0:len(X)//cv]
                y_train = y[len(y)//cv:len(y)]
                y_test = y[0:len(y)//cv]
                train_X = random_test_set4(X_train,y)
                test_X = random_test_set4(X_test,y)
        else:
            X_train = X[0:len(X)//10*i]
            X_train.extend(X[len(X)//10*(i+1):len(X)])
            X_test = X[len(X)//10*i:len(X)//10*(i+1)]
            y_train = y[0:len(y)//10*i]
            y_train.extend(y[len(y)//10*(i+1):len(y)])
            y_test = y[len(y)//10*i:len(y)//10*(i+1)]
            train_X = random_test_set4(X_train,y)
            test_X = random_test_set4(X_test,y)
        
        _ = cv_clf.fit(X_train,y_train)
        predict = cv_clf.predict(X_test)
        predict1 = cv_clf.predict(test_X)
        score = 0
        score1 = 0
        _ = cv_clf2.fit(train_X,y_train)
        predict2 = cv_clf2.predict(test_X)
        predict22 = cv_clf2.predict(X_test)

        score2 = 0
        score22 = 0
        for i in range(0,len(y_test)):
            if y_test[i] == predict[i]:
                score = score + 1
            if y_test[i] == predict1[i]:
                score1 = score1 + 1
            if y_test[i] == predict2[i]:
                score2 = score2 + 1
            if y_test[i] == predict22[i]:
                score22 = score22 + 1
        scorings1.append(score/len(y_test))
        scorings2.append(score1/len(y_test))
        scorings3.append(score2/len(y_test))
        scorings4.append(score22/len(y_test))

#    y_compare = []
#    y_compare.append(y_test)
#    y_compare.append(predict)
#    y_compare.append(predict1)
#    y_compare.append(predict2)
#    y_compare.append(predict22)
    scorings.append(scorings1)
    scorings.append(scorings2)
    scorings.append(scorings3)
    scorings.append(scorings4)
    return scorings

#cv for adding absolute amounts
    # @X features
    # @y target 
    # @clf classifier 
    # @cv amount of splits for the cross validation
    # @amount absolute amount added to features
    # @information amount of returned information
    # @returns scoring a list of scores for each cv part and each tested option 
    # @returns guessed if information > 1, summary of the predictions, split up by the classes in the target
    # @returns predicts predictions for each cv part
def cv_scores_noise2(X,y,clf,cv,amount,information):
    X,y = shuffle_set(X,y) 
    scorings = [[],[],[],[]]
    score = [[],[],[],[]]
    predict = [[],[],[],[]]
    guessed = [[],[],[],[]]
    predicts = [[],[],[],[],[]]
    for i in range(0,cv):
        cv_clf = copy(clf)
        cv_clf2 = copy(clf)
        if i == 0 or i== 9:
            if i== 0 :
                X_train = X[0:len(X)-len(X)//cv]
                X_test = X[len(X)-len(X)//cv:len(X)]
                y_train = y[0:len(y)-len(y)//cv]
                y_test = y[len(y)-len(y)//cv:len(y)]
                if amount == 0:
                    train_X = random_test_set4(X_train,y)
                    test_X = random_test_set4(X_test,y)
                else:
                    train_X = random_test_set3(X_train,amount)
                    test_X = random_test_set3(X_test,amount)
            else:
                X_train = X[len(X)//cv:len(X)]
                X_test = X[0:len(X)//cv]
                y_train = y[len(y)//cv:len(y)]
                y_test = y[0:len(y)//cv]
                if amount == 0:                        
                    train_X = random_test_set4(X_train,y)
                    test_X = random_test_set4(X_test,y)
                else:
                    train_X = random_test_set3(X_train,amount)
                    test_X = random_test_set3(X_test,amount)
        else:
            X_train = X[0:len(X)//10*i]
            X_train.extend(X[len(X)//10*(i+1):len(X)])
            X_test = X[len(X)//10*i:len(X)//10*(i+1)]
            y_train = y[0:len(y)//10*i]
            y_train.extend(y[len(y)//10*(i+1):len(y)])
            y_test = y[len(y)//10*i:len(y)//10*(i+1)]
            if amount == 0:                
                train_X = random_test_set4(X_train,y)
                test_X = random_test_set4(X_test,y)
            else:
                train_X = random_test_set3(X_train,amount)
                test_X = random_test_set3(X_test,amount)
        
        _ = cv_clf.fit(X_train,y_train)
        predict[0] = cv_clf.predict(X_test)
        predict[1] = cv_clf.predict(test_X)
        for k in range(0,4):
            score[k] = 0
        _ = cv_clf2.fit(train_X,y_train)
        predict[2] = cv_clf2.predict(test_X)
        predict[3] = cv_clf2.predict(X_test)
        if information >=2:            
            for k in range(0,4):
                guessed[k].append(distr_guessed(predict[k]))
        for k in range(0,len(y_test)):
            for j in range(0,4):
                if y_test[k] == predict[j][k]:
                    score[j] = score[j] + 1

        for k in range(0,4):
            scorings[k].append(accuracy_score(y_test,predict[k]))
        if information >= 3:
            for k in range(0,4):
                predicts[k].append(predict[k])
            predicts[4].append(y_test)
    if(information >= 3):
        return scorings,guessed,predicts
    elif information == 2:
        return scorings,guessed
    else:
        return scorings
    
    
def cv_scores_noise3(X,y,catagorical,clf,cv,amount,information):
    X,y = shuffle_set(X,y) 
    scorings = [[],[],[],[]]
    score = [[],[],[],[]]
    predict = [[],[],[],[]]
    guessed = [[],[],[],[]]
    predicts = [[],[],[],[],[]]
    for i in range(0,cv):
        cv_clf = copy(clf)
        cv_clf2 = copy(clf)
        if i == 0 or i== 9:
            if i== 0 :
                X_train = X[0:len(X)-len(X)//cv]
                X_test = X[len(X)-len(X)//cv:len(X)]
                y_train = y[0:len(y)-len(y)//cv]
                y_test = y[len(y)-len(y)//cv:len(y)]
                if amount == 0:
                    train_X = random_test_set4(X_train,y)
                    test_X = random_test_set4(X_test,y)
                else:
                    train_X = random_test_set6(X_train,catagorical,amount)
                    test_X = random_test_set6(X_test,catagorical,amount)
            else:
                X_train = X[len(X)//cv:len(X)]
                X_test = X[0:len(X)//cv]
                y_train = y[len(y)//cv:len(y)]
                y_test = y[0:len(y)//cv]
                if amount == 0:                        
                    train_X = random_test_set4(X_train,y)
                    test_X = random_test_set4(X_test,y)
                else:
                    train_X = random_test_set6(X_train,catagorical,amount)
                    test_X = random_test_set6(X_test,catagorical,amount)
        else:
            X_train = X[0:len(X)//10*i]
            X_train.extend(X[len(X)//10*(i+1):len(X)])
            X_test = X[len(X)//10*i:len(X)//10*(i+1)]
            y_train = y[0:len(y)//10*i]
            y_train.extend(y[len(y)//10*(i+1):len(y)])
            y_test = y[len(y)//10*i:len(y)//10*(i+1)]
            if amount == 0:                
                train_X = random_test_set4(X_train,y)
                test_X = random_test_set4(X_test,y)
            else:
                train_X = random_test_set6(X_train,catagorical,amount)
                test_X = random_test_set6(X_test,catagorical,amount)
        
        _ = cv_clf.fit(X_train,y_train)
        predict[0] = cv_clf.predict(X_test)
        predict[1] = cv_clf.predict(test_X)
        for k in range(0,4):
            score[k] = 0
        _ = cv_clf2.fit(train_X,y_train)
        predict[2] = cv_clf2.predict(test_X)
        predict[3] = cv_clf2.predict(X_test)
        if information >=2:            
            for k in range(0,4):
                guessed[k].append(distr_guessed(predict[k]))
        for k in range(0,len(y_test)):
            for j in range(0,4):
                if y_test[k] == predict[j][k]:
                    score[j] = score[j] + 1

        for k in range(0,4):
            scorings[k].append(accuracy_score(y_test,predict[k]))
        if information >= 3:
            for k in range(0,4):
                predicts[k].append(predict[k])
            predicts[4].append(y_test)
    if(information >= 3):
        return scorings,guessed,predicts
    elif information == 2:
        return scorings,guessed
    else:
        return scorings
    
    
def cv_scores_noise4(X,y,clf,cv,amount,information):
    X,y = shuffle_set(X,y) 
    scorings = [[],[],[],[]]
    score = [[],[],[],[]]
    predict = [[],[],[],[]]
    guessed = [[],[],[],[]]
    predicts = [[],[],[],[],[]]
    for i in range(0,cv):
        cv_clf = copy(clf)
        cv_clf2 = copy(clf)
        if i == 0 or i== 9:
            if i== 0 :
                X_train = X[0:len(X)-len(X)//cv]
                X_test = X[len(X)-len(X)//cv:len(X)]
                y_train = y[0:len(y)-len(y)//cv]
                y_test = y[len(y)-len(y)//cv:len(y)]
                if amount == 0:
                    train_X = random_test_set4(X_train,y)
                    test_X = random_test_set4(X_test,y)
                else:
                    train_X = random_test_set7(X_train,amount)
                    test_X = random_test_set7(X_test,amount)
            else:
                X_train = X[len(X)//cv:len(X)]
                X_test = X[0:len(X)//cv]
                y_train = y[len(y)//cv:len(y)]
                y_test = y[0:len(y)//cv]
                if amount == 0:                        
                    train_X = random_test_set4(X_train,y)
                    test_X = random_test_set4(X_test,y)
                else:
                    train_X = random_test_set7(X_train,amount)
                    test_X = random_test_set7(X_test,amount)
        else:
            X_train = X[0:len(X)//10*i]
            X_train.extend(X[len(X)//10*(i+1):len(X)])
            X_test = X[len(X)//10*i:len(X)//10*(i+1)]
            y_train = y[0:len(y)//10*i]
            y_train.extend(y[len(y)//10*(i+1):len(y)])
            y_test = y[len(y)//10*i:len(y)//10*(i+1)]
            if amount == 0:                
                train_X = random_test_set4(X_train,y)
                test_X = random_test_set4(X_test,y)
            else:
                train_X = random_test_set7(X_train,amount)
                test_X = random_test_set7(X_test,amount)
        
        _ = cv_clf.fit(X_train,y_train)
        predict[0] = cv_clf.predict(X_test)
        predict[1] = cv_clf.predict(test_X)
        for k in range(0,4):
            score[k] = 0
        _ = cv_clf2.fit(train_X,y_train)
        predict[2] = cv_clf2.predict(test_X)
        predict[3] = cv_clf2.predict(X_test)
        if information >=2:            
            for k in range(0,4):
                guessed[k].append(distr_guessed(predict[k]))
        for k in range(0,4):
            scorings[k].append(accuracy_score(y_test,predict[k]))
        if information >= 3:
            for k in range(0,4):
                predicts[k].append(predict[k])
            predicts[4].append(y_test)
    if(information >= 3):
        return scorings,guessed,predicts
    elif information == 2:
        return scorings,guessed
    else:
        return scorings
    
def cv_scores_noise4(X,y,clf,cv,amount,information):
    X,y = shuffle_set(X,y) 
    scorings = [[],[],[],[]]
    score = [[],[],[],[]]
    predict = [[],[],[],[]]
    guessed = [[],[],[],[]]
    predicts = [[],[],[],[],[]]
    for i in range(0,cv):
        cv_clf = copy(clf)
        cv_clf2 = copy(clf)
        if i == 0 or i== 9:
            if i== 0 :
                X_train = X[0:len(X)-len(X)//cv]
                X_test = X[len(X)-len(X)//cv:len(X)]
                y_train = y[0:len(y)-len(y)//cv]
                y_test = y[len(y)-len(y)//cv:len(y)]
                if amount == 0:
                    train_X = random_test_set4(X_train,y)
                    test_X = random_test_set4(X_test,y)
                else:
                    train_X = random_test_set7(X_train,amount)
                    test_X = random_test_set7(X_test,amount)
            else:
                X_train = X[len(X)//cv:len(X)]
                X_test = X[0:len(X)//cv]
                y_train = y[len(y)//cv:len(y)]
                y_test = y[0:len(y)//cv]
                if amount == 0:                        
                    train_X = random_test_set4(X_train,y)
                    test_X = random_test_set4(X_test,y)
                else:
                    train_X = random_test_set7(X_train,amount)
                    test_X = random_test_set7(X_test,amount)
        else:
            X_train = X[0:len(X)//10*i]
            X_train.extend(X[len(X)//10*(i+1):len(X)])
            X_test = X[len(X)//10*i:len(X)//10*(i+1)]
            y_train = y[0:len(y)//10*i]
            y_train.extend(y[len(y)//10*(i+1):len(y)])
            y_test = y[len(y)//10*i:len(y)//10*(i+1)]
            if amount == 0:                
                train_X = random_test_set4(X_train,y)
                test_X = random_test_set4(X_test,y)
            else:
                train_X = random_test_set7(X_train,amount)
                test_X = random_test_set7(X_test,amount)
        
        _ = cv_clf.fit(X_train,y_train)
        predict[0] = cv_clf.predict(X_test)
        predict[1] = cv_clf.predict(test_X)
        for k in range(0,4):
            score[k] = 0
        _ = cv_clf2.fit(train_X,y_train)
        predict[2] = cv_clf2.predict(test_X)
        predict[3] = cv_clf2.predict(X_test)
        if information >=2:            
            for k in range(0,4):
                guessed[k].append(distr_guessed(predict[k]))
        
        for k in range(0,4):
            scorings[k].append(accuracy_score(y_test,predict[k]))
        if information >= 3:
            for k in range(0,4):
                predicts[k].append(predict[k])
            predicts[4].append(y_test)
    if(information >= 3):
        return scorings,guessed,predicts
    elif information == 2:
        return scorings,guessed
    else:
        return scorings
    
    
def cv_scores_noise5(X,y,catagorical,clf,cv,amount,information):
    X,y = shuffle_set(X,y) 
    scorings = [[],[],[],[]]
    score = [[],[],[],[]]
    predict = [[],[],[],[]]
    guessed = [[],[],[],[]]
    predicts = [[],[],[],[],[]]
    for i in range(0,cv):
        cv_clf = copy(clf)
        cv_clf2 = copy(clf)
        if i == 0 or i== 9:
            if i== 0 :
                X_train = X[0:len(X)-len(X)//cv]
                X_test = X[len(X)-len(X)//cv:len(X)]
                y_train = y[0:len(y)-len(y)//cv]
                y_test = y[len(y)-len(y)//cv:len(y)]
                if amount > 1:
                    train_X = random_test_set9(X_train,catagorical,amount)
                    test_X = random_test_set9(X_test,catagorical,amount)
                else:
                    train_X = random_test_set8(X_train,catagorical,amount)
                    test_X = random_test_set8(X_test,catagorical,amount)
            else:
                X_train = X[len(X)//cv:len(X)]
                X_test = X[0:len(X)//cv]
                y_train = y[len(y)//cv:len(y)]
                y_test = y[0:len(y)//cv]
                if amount > 1:                        
                    train_X = random_test_set9(X_train,catagorical,amount)
                    test_X = random_test_set9(X_test,catagorical,amount)
                else:
                    train_X = random_test_set8(X_train,catagorical,amount)
                    test_X = random_test_set8(X_test,catagorical,amount)
        else:
            X_train = X[0:len(X)//10*i]
            X_train.extend(X[len(X)//10*(i+1):len(X)])
            X_test = X[len(X)//10*i:len(X)//10*(i+1)]
            y_train = y[0:len(y)//10*i]
            y_train.extend(y[len(y)//10*(i+1):len(y)])
            y_test = y[len(y)//10*i:len(y)//10*(i+1)]
            if amount == 0:                
                train_X = random_test_set9(X_train,catagorical,amount)
                test_X = random_test_set9(X_test,catagorical,amount)
            else:
                train_X = random_test_set8(X_train,catagorical,amount)
                test_X = random_test_set8(X_test,catagorical,amount)
        
        _ = cv_clf.fit(X_train,y_train)
        predict[0] = cv_clf.predict(X_test)
        predict[1] = cv_clf.predict(test_X)
        for k in range(0,4):
            score[k] = 0
        _ = cv_clf2.fit(train_X,y_train)
        predict[2] = cv_clf2.predict(test_X)
        predict[3] = cv_clf2.predict(X_test)
        if information >=2:            
            for k in range(0,4):
                guessed[k].append(distr_guessed(predict[k]))        
                    
        for k in range(0,4):
            scorings[k].append(accuracy_score(y_test,predict[k]))
        if information >= 3:
            for k in range(0,4):
                predicts[k].append(predict[k])
            predicts[4].append(y_test)
    if(information >= 3):
        return scorings,guessed,predicts
    elif information == 2:
        return scorings,guessed
    else:
        return scorings
    
#cv for adding absolute amounts
    # @X features
    # @y target 
    # @clf classifier 
    # @cv amount of splits for the cross validation
    # @amount of noise, ratio or abosulte depending on cvScore
    # @information amount of returned information
    # @cvScore == 2 add an abosulte amount of Noise
    # @cvScore == 3 multiple by amount for symbolic features, categorical get an absolute increase
    # @cvScore == 4 multiple all features by amount
    # @cvScore == 5 symbolic features get multiplied if amount > 1 by (random()+0.5,
    #               for 0< amount <1 > random() for each feature to be multiplied 
    # @returns scoring a list of scores for each cv part and each tested option 
    # @returns guessed if information > 1, summary of the predictions, split up by the classes in the target
    # @returns predicts predictions for each cv part 
def cv_scores_noise(X,y,clf,cv,amount,information,cvScore):
    X,y = shuffle_set(X,y) 
    scorings = [[],[],[],[]]
    score = [[],[],[],[]]
    predict = [[],[],[],[]]
    guessed = [[],[],[],[]]
    predicts = [[],[],[],[],[]]
    for i in range(0,cv):
        cv_clf = copy(clf)
        cv_clf2 = copy(clf)
        X_train,y_train,train_X,test_X = cv_noise_splits(X,y,i,cvScore)        
        _ = cv_clf.fit(X_train,y_train)
        predict[0] = cv_clf.predict(X_test)
        predict[1] = cv_clf.predict(test_X)
        for k in range(0,4):
            score[k] = 0
        _ = cv_clf2.fit(train_X,y_train)
        predict[2] = cv_clf2.predict(test_X)
        predict[3] = cv_clf2.predict(X_test)
        if information >=2:            
            for k in range(0,4):
                guessed[k].append(distr_guessed(predict[k]))
        for k in range(0,len(y_test)):
            for j in range(0,4):
                if y_test[k] == predict[j][k]:
                    score[j] = score[j] + 1

        for k in range(0,4):
            scorings[k].append(accuracy_score(y_test,predict[k]))
        if information >= 3:
            for k in range(0,4):
                predicts[k].append(predict[k])
            predicts[4].append(y_test)
    if(information >= 3):
        return scorings,guessed,predicts
    elif information == 2:
        return scorings,guessed
    else:
        return scorings
 
    
def cv_noise_splits(X,y,i, cvScore):
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
    return X_train,y_train,train_X,test_X
#Remove features
#amount is factor
def cv_scores_features1(X,y,clf,cv,amount,information):
    dur = 0         
    X,y = shuffle_set(X,y)
#    amount = round(len(X[0])*amount)
    sc = 2
    scorings = [[],[]]
    score = [[],[]]
    predict = [[],[]]
    guessed = [[],[]]
    predicts = [[],[],[]]
    time = [0,0,0,0]
    for i in range(0,cv):
        cv_clf = copy(clf)
        cv_clf2 = copy(clf)
        if i == 0 or i== 9:
            if i== 0 :
                X_train = X[0:len(X)-len(X)//cv]
                X_test = X[len(X)-len(X)//cv:len(X)]
                y_train = y[0:len(y)-len(y)//cv]
                y_test = y[len(y)-len(y)//cv:len(y)]
                train_X = remove_features(X_train,amount)
                test_X = remove_features(X_test,amount)
            else:
                X_train = X[len(X)//cv:len(X)]
                X_test = X[0:len(X)//cv]
                y_train = y[len(y)//cv:len(y)]
                y_test = y[0:len(y)//cv]
                train_X = remove_features(X_train,amount)
                test_X = remove_features(X_test,amount)
        else:
            X_train = X[0:len(X)//10*i]
            X_train.extend(X[len(X)//10*(i+1):len(X)])
            X_test = X[len(X)//10*i:len(X)//10*(i+1)]
            y_train = y[0:len(y)//10*i]
            y_train.extend(y[len(y)//10*(i+1):len(y)])
            y_test = y[len(y)//10*i:len(y)//10*(i+1)]
            train_X = remove_features(X_train,amount)
            test_X = remove_features(X_test,amount)
        with stopwatch() as sw:
            _ = cv_clf.fit(X_train,y_train)
        time[0] = time[0] + sw.duration
        with stopwatch() as sw:
            predict[0] = cv_clf.predict(X_test)
        time[1] = time[1] + sw.duration
        for k in range(0,sc):
            score[k] = 0
        with stopwatch() as sw:    
            _ = cv_clf2.fit(train_X,y_train)
        time[2] = time[2] + sw.duration
        with stopwatch() as sw:
            predict[1] = cv_clf2.predict(test_X)
        time[3] = time[3] + sw.duration
        if information >=2:            
            for k in range(0,sc):
                guessed[k].append(distr_guessed(predict[k]))        
                    
        for k in range(0,sc):
            scorings[k].append(accuracy_score(y_test,predict[k]))
        if information >= 3:
            for k in range(0,sc):
                predicts[k].append(predict[k])
            predicts[sc].append(y_test)
        
    
    if(information >= 3):
        return scorings,guessed,predicts,time
    
    elif information == 2:
        return scorings,guessed
    else:
        return scorings

    
#adding features    
def cv_scores_features3(X,y,clf,cv,amount,information):
    dur = 0         
    X,y = shuffle_set(X,y)
    sc = 2
    scorings = [[],[]]
    score = [[],[]]
    predict = [[],[]]
    guessed = [[],[]]
    predicts = [[],[],[]]
    time = [0,0,0,0]
    for i in range(0,cv):
        cv_clf = copy(clf)
        cv_clf2 = copy(clf)
        if i == 0 or i== 9:
            if i== 0 :
                X_train = X[0:len(X)-len(X)//cv]
                X_test = X[len(X)-len(X)//cv:len(X)]
                y_train = y[0:len(y)-len(y)//cv]
                y_test = y[len(y)-len(y)//cv:len(y)]
                train_X = add_noise_features(X_train,amount)
                test_X = add_noise_features(X_test,amount)
            else:
                X_train = X[len(X)//cv:len(X)]
                X_test = X[0:len(X)//cv]
                y_train = y[len(y)//cv:len(y)]
                y_test = y[0:len(y)//cv]
                train_X = add_noise_features(X_train,amount)
                test_X = add_noise_features(X_test,amount)
        else:
            X_train = X[0:len(X)//10*i]
            X_train.extend(X[len(X)//10*(i+1):len(X)])
            X_test = X[len(X)//10*i:len(X)//10*(i+1)]
            y_train = y[0:len(y)//10*i]
            y_train.extend(y[len(y)//10*(i+1):len(y)])
            y_test = y[len(y)//10*i:len(y)//10*(i+1)]
            train_X = add_noise_features(X_train,amount)
            test_X = add_noise_features(X_test,amount)
        with stopwatch() as sw:
            _ = cv_clf.fit(X_train,y_train)
        time[0] = time[0] + sw.duration
        with stopwatch() as sw:
            predict[0] = cv_clf.predict(X_test)
        time[1] = time[1] + sw.duration
        for k in range(0,sc):
            score[k] = 0
        with stopwatch() as sw:    
            _ = cv_clf2.fit(train_X,y_train)
        time[2] = time[2] + sw.duration
        with stopwatch() as sw:
            predict[1] = cv_clf2.predict(test_X)
        time[3] = time[3] + sw.duration
        if information >=2:            
            for k in range(0,sc):
                guessed[k].append(distr_guessed(predict[k]))        
                    
        for k in range(0,sc):
            scorings[k].append(accuracy_score(y_test,predict[k]))
        if information >= 3:
            for k in range(0,sc):
                predicts[k].append(predict[k])
            predicts[sc].append(y_test)
        
    
    if(information >= 3):
        return scorings,guessed,predicts,time
    
    elif information == 2:
        return scorings,guessed
    else:
        return scorings
    
def cv_scores_features4(X,y,cat,clf,cv,amount,information):        
    X,y = shuffle_set(X,y)
    sc = 2
    scorings = [[],[]]
    score = [[],[]]
    predict = [[],[]]
    guessed = [[],[]]
    predicts = [[],[],[]]
    time = [0,0,0,0]
    for i in range(0,cv):
        cv_clf = copy(clf)
        cv_clf2 = copy(clf)
        if i == 0 or i== 9:
            if i== 0 :
                X_train = X[0:len(X)-len(X)//cv]
                X_test = X[len(X)-len(X)//cv:len(X)]
                y_train = y[0:len(y)-len(y)//cv]
                y_test = y[len(y)-len(y)//cv:len(y)]
                train_X = add_noise_features2(X_train,cat,amount)
                test_X = add_noise_features2(X_test,cat,amount)
            else:
                X_train = X[len(X)//cv:len(X)]
                X_test = X[0:len(X)//cv]
                y_train = y[len(y)//cv:len(y)]
                y_test = y[0:len(y)//cv]
                train_X = add_noise_features2(X_train,cat,amount)
                test_X = add_noise_features2(X_test,cat,amount)
        else:
            X_train = X[0:len(X)//10*i]
            X_train.extend(X[len(X)//10*(i+1):len(X)])
            X_test = X[len(X)//10*i:len(X)//10*(i+1)]
            y_train = y[0:len(y)//10*i]
            y_train.extend(y[len(y)//10*(i+1):len(y)])
            y_test = y[len(y)//10*i:len(y)//10*(i+1)]
            train_X = add_noise_features2(X_train,cat,amount)
            test_X = add_noise_features2(X_test,cat,amount)
        with stopwatch() as sw:
            _ = cv_clf.fit(X_train,y_train)
        time[0] = time[0] + sw.duration
        with stopwatch() as sw:
            predict[0] = cv_clf.predict(X_test)
        time[1] = time[1] + sw.duration
        for k in range(0,sc):
            score[k] = 0
        with stopwatch() as sw:    
            _ = cv_clf2.fit(train_X,y_train)
        time[2] = time[2] + sw.duration
        with stopwatch() as sw:
            predict[1] = cv_clf2.predict(test_X)
        time[3] = time[3] + sw.duration
        if information >=2:            
            for k in range(0,sc):
                guessed[k].append(distr_guessed(predict[k]))        
                    
        for k in range(0,sc):
            scorings[k].append(accuracy_score(y_test,predict[k]))
        if information >= 3:
            for k in range(0,sc):
                predicts[k].append(predict[k])
            predicts[sc].append(y_test)
        
    
    if(information >= 3):
        return scorings,guessed,predicts,time
    
    elif information == 2:
        return scorings,guessed
    else:
        return scorings
    
    
    

def cv_scores_missing(X,y,cat,clf,cv,amount,information):        
    X,y = shuffle_set(X,y)
    sc = 2
    scorings = [[],[]]
    score = [[],[]]
    predict = [[],[]]
    guessed = [[],[]]
    predicts = [[],[],[]]
    time = [0,0,0]
    for i in range(0,cv):
        cv_clf = copy(clf)
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
            
        test_X = test_set_missing_values(X_test,amount)
        with stopwatch() as sw:
            _ = cv_clf.fit(X_train,y_train)
        time[0] = time[0] + sw.duration
        with stopwatch() as sw:
            predict[0] = cv_clf.predict(X_test)
        time[1] = time[1] + sw.duration
        for k in range(0,sc):
            score[k] = 0
        with stopwatch() as sw:
            predict[1] = cv_clf.predict(test_X)
        time[2] = time[2] + sw.duration
        if information >=2:            
            for k in range(0,sc):
                guessed[k].append(distr_guessed(predict[k]))        
                    
        for k in range(0,sc):
            scorings[k].append(accuracy_score(y_test,predict[k]))
        if information >= 3:
            for k in range(0,sc):
                predicts[k].append(predict[k])
            predicts[sc].append(y_test)
        
    
    if(information >= 3):
        return scorings,guessed,predicts,time
    
    elif information == 2:
        return scorings,guessed
    else:
        return scorings
    
    
def cv_scores_scales(X,y,clf,cv,amount,information):        
    X,y = shuffle_set(X,y)
    sc = 1
    scorings = [[]]
    score = [[]]
    predict = [[]]
    guessed = [[]]
    predicts = [[],[]]
    time = [0,0]
    X,y  = split(X,y,amount)
    for i in range(0,cv):
        cv_clf = copy(clf)
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
        with stopwatch() as sw:
            _ = cv_clf.fit(X_train,y_train)
        time[0] = time[0] + sw.duration
        with stopwatch() as sw:
            predict[0] = cv_clf.predict(X_test)
        time[1] = time[1] + sw.duration
        for k in range(0,sc):
            score[k] = 0
        if information >=2:            
            for k in range(0,sc):
                guessed[k].append(distr_guessed(predict[k]))        
                    
        for k in range(0,sc):
            scorings[k].append(accuracy_score(y_test,predict[k]))
        if information >= 3:
            for k in range(0,sc):
                predicts[k].append(predict[k])
            predicts[sc].append(y_test)
        
    
    if(information >= 3):
        return scorings,guessed,predicts,time
    
    elif information == 2:
        return scorings,guessed
    else:
        return scorings