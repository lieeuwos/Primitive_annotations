# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:17:19 2017

@author: S127788
"""

import csv
from preamble import *
import os
import ast
from sklearn.metrics import accuracy_score,precision_score
from sklearn.metrics import recall_score,zero_one_loss,cohen_kappa_score

pathL = 
dropbox = 

def download_save_sets(list):
    for i in list:
        data = oml.datasets.get_dataset(i)
        X, y, categorical = data.get_data(target = data.default_target_attribute, return_categorical_indicator = True)
        with open(pathL + 'did\\' + str(i) + 'X.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(X)
        with open(pathL + 'did\\' + str(i) + 'y.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows([y])
        with open(pathL + 'did\\' + str(i) + 'cat.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows([categorical])
    
def read_did(did):
    with open(pathL + 'did\\' + str(did) + 'X.csv', 'r') as f:    
        reader2 = csv.reader(f)
        your_list_X = list(reader2)
    for i in range(0,len(your_list_X)):
       for j in range(0,len(your_list_X[i])):
           your_list_X[i][j] = float(your_list_X[i][j] )
    new_list_X = []
    for i in range(0,len(your_list_X)//2):
        new_list_X.append(your_list_X[i*2])
    with open(pathL + 'did\\' + str(did) + 'y.csv', 'r') as f:    
        reader2 = csv.reader(f)
        your_list_y = list(reader2)
    for i in range(0,len(your_list_y)):
       for j in range(0,len(your_list_y[i])):
           your_list_y[i][j] = ast.literal_eval(your_list_y[i][j] )
    new_list_y = your_list_y[0]
    return new_list_X,new_list_y


def read_did_cat(did):
    with open(pathL + 'did\\' + str(did) + 'cat.csv', 'r') as f:    
        reader2 = csv.reader(f)
        your_list_cat = list(reader2)
    for i in range(0,len(your_list_cat)):
       for j in range(0,len(your_list_cat[i])):
           your_list_cat[i][j] = (check_bool(your_list_cat[i][j]))
    new_list_cat = your_list_cat[0]
    return new_list_cat

def check_bool(string):
    if (string == "False"):
        return False
    else: 
        return True
    
def saveDictsDatasets(dicts,func,clfName,amount,did):
    assert type(name) == str
    newpath = dropbox + func + '\\' + clfName + '\\' + str(amount) + '\\' + str(did) + '\\'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    for i in dicts:
        newpath = dropbox + str(i) + '\\'
        with open(newpath + name + '.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(dicts[i])
            
def saveSingleDict(dict1,func,clfName,amount,did,name):
    assert type(name) == str
    assert type(func) == str
    assert type(clfName) == str
    newpath = dropbox + func + '\\' + clfName + '\\' + str(amount) + '\\' + str(did) + '\\'
    if not os.path.exists(newpath):
        os.makedirs(newpath)    
    with open(newpath + name + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(dict1)
        
def saveEstimator(str1,func,clfName,amount,did,name):
    assert type(name) == str
    assert type(func) == str
    assert type(clfName) == str
    newpath = dropbox + func + '\\' + clfName + '\\' + str(amount) + '\\' + str(did) + '\\'
    if not os.path.exists(newpath):
        os.makedirs(newpath)    
    with open(newpath + name + '.txt','a+') as f: 
        f.write(str1)
        
def saveDuration(dict1,func,clfName,amount,did,name):
    assert type(name) == str
    assert type(func) == str
    assert type(clfName) == str
    newpath = dropbox + func + '\\' + clfName + '\\' + str(amount) + '\\' + str(did) + '\\'
    if not os.path.exists(newpath):
        os.makedirs(newpath) 
    with open(newpath + name + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(dict1)
        
def saveDict(dict1,did):
    newpath = dropbox + '\\did\\'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    with open(newpath + str(did) + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows([[dict1]])
        
def readDict(did):
    newpath = dropbox + '\\did\\'
    with open(newpath + str(did) + '.csv', 'r') as f:
        reader2 = csv.reader(f)
        your_dict = list(reader2)
    new_dict = []
    for i in your_dict:
        if not i == []:
            new_dict.append([])
            for j in i:
                new_dict[len(new_dict)-1].append(ast.literal_eval(j))
    new_dict = new_dict[0][0] 
    return new_dict
    
      
        
def savePredictsScore(dict1,func,clfName,amount,did,name):
    counter = 0
    for i in dict1:
        counter = counter + 1
        saveSingleDict(i,func,clfName,amount,did,name + str(counter))
        
def read_did_pred(func,clfName,amount,did,name):
    assert type(name) == str
    assert type(func) == str
    assert type(clfName) == str
    if checkForExist(func,clfName,amount,did):
        newpath = dropbox + func + '\\' + clfName + '\\' + str(amount) + '\\' + str(did) + '\\'
        
        with open(newpath + name + '.csv', 'r') as f:    
            reader2 = csv.reader(f)
            your_dict = list(reader2)
        new_dict = []
        for i in your_dict:
            if not i == []:
                new_dict.append([])
                for j in i:
                    new_dict[len(new_dict)-1].append(ast.literal_eval(j))
        return new_dict
    else:
        print('missing' + func+ ',' + clfName+ ',' + str(amount) + ','  + str(did)+ ',' + name)
        return 0



def read_duration(func,clfName,amount,did):
    duration = [] # amount of scores in the set
    name = 'duration'
    Ramount = checkForExistFile(func,clfName,amount,did)
    for i in range(0,Ramount):
        temp = read_did_pred(func,clfName,amount,did,name + str(i))[0]
        for j in range(0,len(temp)):
            duration.append(0)
            duration[j] = duration[j] + temp[j]/Ramount
    return duration

def read_features(func,clfName,amount,did):
    scorez = [0,0,0] # amount of scores in the set
    name = 'scores'
    Ramount = checkForExistFile(func,clfName,amount,did)
    for i in range(0,Ramount):
        temp = read_did_pred(func,clfName,amount,did,name + str(i))
        for i in range(0,len(temp)):
            scorez[i] = scorez[i] + sum(temp[i])/len(temp[i])/Ramount
    return scorez

def read_did_preds(func,clfName,amount,did,name):
    list1 = []
    for i in range(1,6):
        list1.append(read_did_pred(func,clfName,amount,did,name + str(i)))
    return list1
        
def read_did_predSummary(func,clfName,amount,did,name):
    assert type(name) == str
    assert type(func) == str
    assert type(clfName) == str
    newpath = dropbox + func + '\\' + clfName + '\\' + str(amount) + '\\' + str(did) + '\\'

    with open(newpath + name + '.csv', 'r') as f:    
        reader2 = csv.reader(f)
        your_dict = list(reader2)
    new_dict = []
    for i in your_dict:
        if not i == []:
            new_dict.append([])
            for j in i:
                new_dict[len(new_dict)-1].append(ast.literal_eval(j))
    return new_dict

def checkForExist(func,clfName,amount,did):          
    path = dropbox + func + '\\' + clfName + '\\' + str(amount) + '\\' + str(did)
    return os.path.isdir(path)

def checkForExistFile(func,clfName,amount,did):          
    for i in range(0,100):
        path = dropbox + func + '\\' + clfName + '\\' + str(amount) + '\\' + str(did) + '\\scores'
        path = path + str(i) + '.csv'
        if not os.path.isfile(path):
            return i
    return 0

def ScoresFromPredictions(func,clfName,amount,did,scoreM):
    predictions = read_did_preds(func,clfName,amount,did,'Predictions')
    score = []
    for i in range(0,4):
        score.append([])
        for j in range(0,len(predictions[i])):
            if scoreM == 'accuracy':
                score[i].append(accuracy_score(predictions[i][j],predictions[4][j]))
            elif scoreM == 'precision' and readDict(did)['NumberOfClasses'] == 2:
                score[i].append(precision_score(predictions[i][j],predictions[4][j]))
            elif scoreM == 'recall' and readDict(did)['NumberOfClasses'] == 2:
                score[i].append(recall_score(predictions[i][j],predictions[4][j])) 
            elif scoreM == 'zero_one_loss':
                score[i].append(zero_one_loss(predictions[i][j],predictions[4][j]))
            elif scoreM == 'kappa':
                score[i].append(cohen_kappa_score(predictions[i][j],predictions[4][j]))
        
    return score

def readEstimator(func,clfName,amount,did):
    estimator = []
    name = 'Estimators' 
    Ramount = checkForExistFile(func,clfName,amount,did) +1 
    for i in range(0,Ramount):
        temp = readSingleEstimator(func,clfName,amount,did,name + str(i))
        
        estimator.append(temp)
    return estimator

def readSingleEstimator(func,clfName,amount,did,name):
    newpath = dropbox + func + '\\' + clfName + '\\' + str(amount) + '\\' + str(did) + '\\'
    with open(newpath + name + '.txt','r') as f: 
        output = f.read()
    firstSplit = output.split(', ' + clfFronts(clfName))
    output = []
    for i,x in enumerate(firstSplit):
        if not i==0:
            output.append(clfFronts(clfName) + x)
        else:
            output.append(x)
    return output

def ClfAnalysis(listE,clfName):
    listP = []      
    for j,zu in enumerate(listE):
        if (clfName == 'RandomForestClassifier'):
            for i,x in enumerate(listE[j]):
                tempD = {}
                tempD['max_features'] = float(x.split('max_features=')[1].split(',')[0])
                tempD['min_samples_split'] = float(x.split('min_samples_split=')[1].split(',')[0])
                listP.append(tempD)
        elif (clfName == 'KNeighborsClassifier'):
            for i,x in enumerate(listE[j]):
                tempD = {}
                tempD['weights'] = str(x.split('weights=')[1].split(',')[0])
                tempD['n_neighbors'] = float(x.split('n_neighbors=')[1].split(',')[0])
                tempD['p'] = int(listE[1].split('p=')[i].split(',')[0])
                listP.append(tempD)
        elif (clfName == 'AdaBoost'):
            for i,x in enumerate(listE[j]):
                tempD = {}
                tempD['learning_rate'] = float(x.split('learning_rate=')[1].split(',')[0])
                listP.append(tempD)
        elif (clfName[:4] == 'SVC-'):
            for i,x in enumerate(listE[j]):
                tempD = {}
                tempD['C'] = float(x.split('C=')[1].split(',')[0])
                tempD['gamma'] = float(x.split('gamma=')[1].split(',')[0])
                tempD['kernel'] = str(x.split('kernel=')[1].split(',')[0])
                listP.append(tempD)    
        elif (clfName == 'GradientBoost'):
            for i,x in enumerate(listE[j]):
                tempD = {}
                tempD['learning_rate'] = float(x.split('learning_rate=')[1].split(',')[0])
                listP.append(tempD)
    return listP

def splitClfAnalysis(estimators,cv,splits):
    splitList = []
    for i in range(splits):
        splitList.append([])
    divider = splits - 1
    for i,x in enumerate(estimators):
        if i % cv == 0:
            divider = (divider + 1) % splits
        splitList[divider].append(x)
    return splitList

def clfFronts(clfName):
    if (clfName == 'RandomForestClassifier'):
        front = 'RandomForestClassifier('
    elif (clfName == 'KNeighborsClassifier'):
        front = 'KNeighborsClassifier('
    elif (clfName == 'AdaBoost'):
        front = 'AdaBoostClassifier(' 
    elif (clfName[:4] == 'SVC-'):
        front = 'SVC(' 
    elif (clfName == 'GradientBoost'):
        front = 'GradientBoostClassifier('
    return front