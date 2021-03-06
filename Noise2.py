# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:27:44 2017

@author: S127788
"""
import random
from copy import copy
from copy import deepcopy
from random import shuffle
import numpy as np
from LocalDatasets import readDict
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_mutual_info_score

def add_noise(X,y,adj,local):
    assert len(X) == len(y), "There should be equal feature set as targets"
    assert len(X) >= 10, "There are more than 10 values, enough values"
    return add_noise_amount(X,y,10,adj)#Everything seems fine, carry on

def add_noise_amount(X,y,amount,adj):
    amount = len(y)//amount
    assert len(X) == len(y), "There should be equal feature set as targets"
    assert type(amount) == int , "amount should be integer"
    rand = True
    noise_X = generate_X_noise(X,amount,rand)    
    y_noise = generate_y_noise(y,amount)
    return noise_X, y_noise

def values_target(y):
    set1 = set()
    for i in y:
        if (i not in set1):
            set1.add(i)  
    return set1 #possible values

def weighted_target(y):
    values = dict()
    set1 = set()
    for i in y:
        if values.get(str(i)) == None:
            values[str(i)] = 1
            set1.add(i)
        else:
            values[str(i)] = values[str(i)] + 1
    return values,set1 #possible values

def distr_guessed(y):
    values = dict()
    for i in y:
        if values.get(str(i)) == None:
            values[str(i)] = 1
        else:
            values[str(i)] = values[str(i)] + 1
    return values #possible values



def generate_y_noise(y,amount):
    noise_y = copy(y)
    values, set1 = weighted_target(y)
    for i in set1:
        values[str(i)] = values.get(str(i))*amount/len(y)
    while total_dict(set1,values) < amount:
        temp = str(random.choice(list(set1)))
        values[temp] = values[temp] + 1
    if len(set1) == len(y):
        for i in set1:
            noise_y.append(i)
    else:
        first = random.choice(list(set1))
        noise_y.append(first)
        values[str(first)] = values.get(str(first)) - 1
        for i in range(1,amount):
            first = random.choice(list(set1))
            noise_y.append(first)
            values[str(first)] = values.get(str(first)) - 1
            if values.get(str(first)) == 0:
                set1.remove(first)
    return noise_y #noise for the target

def generate_noise_y(y,amount):
    noise_y = []
    values, set1 = weighted_target(y)
    for i in set1:
        values[str(i)] = values.get(str(i))*amount/len(y)
    while total_dict(set1,values) < amount:
        temp = str(random.choice(list(set1)))
        values[temp] = values[temp] + 1
    if len(set1) == len(y):
        for i in set1:
            noise_y.append(i)
    else:
        first = random.choice(list(set1))
        noise_y.append(first)
        values[str(first)] = values.get(str(first)) - 1
        for i in range(1,amount):
            first = random.choice(list(set1))
            noise_y.append(first)
            values[str(first)] = values.get(str(first)) - 1
            if values.get(str(first)) == 0:
                set1.remove(first)
    return noise_y #noise for the target


def generate_X_noise(X,amount,rand):
    noise_X = copy(X)
    if rand:
        for i in range(0,amount):
            noise_X.append(len(X[0])*[0])
    else:
        for i in range(0,amount):
            noise_X.append(len(X[0])*random())
    return noise_X

def add_noise_features(X,amount):
#    assert amount >= 1, "features should be added"
    assert type(amount) == int , "amount should be integer"
    temp = deepcopy(X)
    for i in temp:
        for j in range(0,amount):
            i.append(random())            
    return temp

def add_copy_features(X,amount):
    temp = deepcopy(X)
    Xt = list(map(list, zip(*X)))
    feature = []
    for i in range(amount):
        feature.append(random.choice(range(len(Xt))))
    for x,i in enumerate(temp):
        for j in range(0,amount):
            i.append(Xt[feature[j]][x])
    return temp


def insert_copy_features(X,amount):
    temp = deepcopy(X)
    Xt = list(map(list, zip(*X)))
    feature = []
    for i in range(amount):
        feature.append(random.choice(range(len(Xt))))
    for x,i in enumerate(temp):
        for j in range(0,amount):
            i.insert(0,Xt[feature[j]][x])
    return temp

def add_copy_featuresCat(X,cat,amount):
    cat2 = copy(cat)
    temp = deepcopy(X)
    Xt = list(map(list, zip(*X)))
    feature = []
    for i in range(amount):        
        feature.append(random.choice(range(len(Xt))))
    for x,i in enumerate(temp):
        for j in range(0,amount):
            i.append(Xt[feature[j]][x])
    for i in feature:
        cat2.append(cat[i])
    return temp,cat2


def add_copy(X,amount):
    temp = deepcopy(X)
    Xt = list(map(list, zip(*X)))
    feature = []
    for i in range(amount):
        feature.append(i)
    for x,i in enumerate(temp):
        for j in range(0,amount):
            i.append(Xt[feature[j]][x])
    return temp

def add_noise_features2(X,cat,amount):
#    assert amount >= 1, "features should be added"
    assert type(amount) == int , "amount should be integer"
    temp = deepcopy(X)
    cats = cat_needed(X,cat,amount)               
    for i in temp:
        for j in range(0,amount):
            if cats > 0:
                i.append(int(100*random.random()))
                cats = cats - 1
            else:
                i.append(random.random())
        cats = cat_needed(X,cat,amount)
    return temp

def add_noise_features2Cat(X,cat,amount):
#    assert amount >= 1, "features should be added"
    assert type(amount) == int , "amount should be integer"
    temp = deepcopy(X)
    cats = cat_needed(X,cat,amount)               
    for i in temp:
        for j in range(0,amount):
            if cats > 0:
                i.append(int(100*random.random()))
                cats = cats - 1
            else:
                i.append(random.random())
        cats = cat_needed(X,cat,amount)
    return temp,balance(cat,X)



def add_noise_features3(X,cat,amount):
#    assert amount >= 1, "features should be added"
    assert type(amount) == int , "amount should be integer"
    temp = deepcopy(X)
    cats = cat_needed(X,cat,amount)
    for i in temp:
        for j in range(0,amount):
            if cats > 0:
                i.append(int(abs(random.gauss(50,50))))
                cats = cats - 1
            else:
                i.append(abs(random.gauss(0,1)))
        cats = cat_needed(X,cat,amount)
                     
    return temp

def cat_needed(X,cat,amount):
    count = 0
    for i in cat:
        if i:
            count = count + 1
    return count

def cat_needed2(X,cat,amount):
    count = 0
    for i in cat:
        if i:
            count = count + 1
            
    return round(amount*(count/len(cat)))
    

def random_test_set(X,y,amount,rand):
    feats = X_feature_examples(X,rand)
    test_X = []
    X_min,X_max = X_features_min_max(X)
    for i in range(0,amount):
        test_X.append(feats)
    test_y = generate_noise_y(y,amount)
    return test_X,test_y

def random_test_set2(X,y,amount):
    X_min,X_max = X_features_min_max(X)
    test_X = []
    for i in range(0,amount):
        temp = []
        for j in range(0,len(X_min)):
            temp.append(random.uniform(X_min,X_max))
        test_X.append(temp)
    return 0
    
def random_test_set3(X,amount):
    noise_X = deepcopy(X)
    #noise_y = copy(y)
    for i in range(0,len(noise_X)):
        for j in range(0,len(noise_X[i])):
            noise_X[i][j] = noise_X[i][j] + amount
    return noise_X

def random_test_set5(X,catagorical,amount):
    noise_X = deepcopy(X)
    #noise_y = copy(y)
    for i in range(0,len(noise_X)):
        for j in range(0,len(noise_X[i])):
            if (catagorical[j]):
                noise_X[i][j] = noise_X[i][j] + int(amount)
            else:
                noise_X[i][j] = noise_X[i][j] + amount
            
    return noise_X

def random_test_set6(X,catagorical,amount):
    noise_X = deepcopy(X)
    #noise_y = copy(y)
    for i in range(0,len(noise_X)):
        for j in range(0,len(noise_X[i])):
            if (catagorical[j]):
                if amount >= 1:
                    noise_X[i][j] = noise_X[i][j] + int(amount)                
            else:
                noise_X[i][j] = noise_X[i][j] * amount
            
    return noise_X




def random_test_set7(X,amount):
    noise_X = deepcopy(X)
    #noise_y = copy(y)
    for i in range(0,len(noise_X)):
        for j in range(0,len(noise_X[i])):
            noise_X[i][j] = noise_X[i][j] * amount
            
    return noise_X

def random_test_set8(X,cat,amount):
    noise_X = deepcopy(X)
    for i in range(0,len(noise_X)):
        for j in range(0,len(noise_X[i])):
            if random() < amount and cat[j] == False:
                noise_X[i][j] = noise_X[i][j] * (random() + 0.5)
            
            
    return noise_X

def random_test_set9(X,cat,amount):
    noise_X = deepcopy(X)
    for i in range(0,len(noise_X)):
        for j in range(0,len(noise_X[i])):
            noise_X[i][j] = noise_X[i][j] * ( amount * (1 + random()))     
            
    return noise_X

def random_test_set4(X,y):
    noise_X = deepcopy(X)
    X_min,X_max = X_features_min_max(X)
    #noise_y = copy(y)
    for i in range(0,len(noise_X)):
        for j in range(0,len(noise_X[i])):
            noise_X[i][j] = noise_X[i][j] + y[i]
    return noise_X

def noise_set(X,cat,amount):
    noise_X = deepcopy(X)
    std_x = std_G(X,cat)
    for i,x in enumerate(noise_X):
        for j,x2 in enumerate(x):
            if random.random() > 0.5 and not cat[j]:
                noise_X[i][j] = noise_X[i][j] + amount*random()*std_x[j]
            elif not cat[j]:
                noise_X[i][j] = noise_X[i][j] - amount*random()*std_x[j]
            else:
                r = random()
                su = 0
                for i in std_x[j]:
                    su = su + std_x[j][str(i)]/len(noise_X)
                    if r > su:
                        noise_X[i][j] = i           
    return noise_X

def noise_set2(X,cat,amount):
    noise_X = deepcopy(X)
    std_x = std_G2(X,cat)
    for i,x in enumerate(noise_X):
        for j,x2 in enumerate(x):
            if random.random() > 0.5 and not cat[j]:
                noise_X[i][j] = noise_X[i][j] + amount*random.random()*std_x[j]
            elif not cat[j]:
                noise_X[i][j] = noise_X[i][j] - amount*random.random()*std_x[j]
            else:
                if random.random()*amount > 0.5:
                    noise_X[i][j] = random.choice(std_x[j])          
    return noise_X

def noise_for_datasets(XList,cat,amount):
    X = XList[0]
    noise_X = deepcopy(X)
    std_x = std_G2(X,cat)
    for i,x in enumerate(noise_X):
        for j,x2 in enumerate(x):
            if random.random() > 0.5 and not cat[j]:
                noise_X[i][j] = noise_X[i][j] + amount*random.random()*std_x[j]
            elif not cat[j]:
                noise_X[i][j] = noise_X[i][j] - amount*random.random()*std_x[j]
            else:
                if random.random()*amount > 0.5:
                    noise_X[i][j] = random.choice(std_x[j])  
    noise_X2 = deepcopy(XList[1])
    for i,x in enumerate(noise_X2):
        for j,x2 in enumerate(x):
            if random.random() > 0.5 and not cat[j]:
                noise_X2[i][j] = noise_X2[i][j] + amount*random.random()*std_x[j]
            elif not cat[j]:
                noise_X2[i][j] = noise_X2[i][j] - amount*random.random()*std_x[j]
            else:
                if random.random()*amount > 0.5:
                    noise_X[i][j] = random.choice(std_x[j])  
    return noise_X,noise_X2

def std_G(X,cat):
    std = []
    Xt = list(map(list, zip(*X)))
    if type(Xt) == np.ndarray:
        for x,i in enumerate(Xt):
            if cat[x]:
                temp1,temp2 = weighted_target(i)
                std.append(temp1)
            else:
                std.append(i.std())
    else:
        for x,i in enumerate(Xt):
            if cat[x]:
                temp1,temp2 = weighted_target(i)
                std.append(temp1)
            else:
                std.append(np.array(i).std())            
    return std

def std_G2(X,cat):
    std = []
    Xt = list(map(list, zip(*X)))
    if type(Xt) == np.ndarray:
        for x,i in enumerate(Xt):
            if cat[x]:
                std.append(i)
            else:
                std.append(i.std())
    else:
        for x,i in enumerate(Xt):
            if cat[x]:
                std.append(i)
            else:
                std.append(np.array(i).std())            
    return std

def test_set_missing_values(X_test,amount):
    X2 = deepcopy(X_test)
    X_Ttrans = list(map(list, zip(*X2)))
    Missings = []
    for i in range(0,amount):
        Missings.append(max(X_Ttrans[i])*100)
    
    for i in range(0,len(X2)):
        for j in range(0,amount):
            X2[i][j] = Missings[j]
    return X2
    
    

def X_feature_examples(X,rand):
    X_trans = list(map(list, zip(*X)))
    feats_X = []
    feats_X_rand = []
    for i in X_trans:
        feats_X.append(sum(i)/len(i))
        feats_X_rand.append(random.uniform(min(i),max(i)))
    if rand:
        return feats_X_rand
    else:
        return feats_X
    
def X_features_min_max(X):
    X_trans = list(map(list, zip(*X)))
    X_min = []
    X_max = []
    for i in X_trans:
        X_min.append(min(i))
        X_max.append(max(i))
    return X_min,X_max

def shuffle_set(X,y):
    shuX = copy(X)
    shuy = copy(y)
    for i in range(0,len(shuy)):
        shuX[i].append(shuy[i])
    shuffle(shuX)
    shuy = []
    for i in range(0,len(shuX)):
        shuy.append(shuX[i].pop())
    return shuX,shuy

    
def zeroList(m,n):
    z_list = []
    for i in range(0,m):
        z_list.append([0]*n)
    return z_list


def total_dict(set1,values):
    assert len(set1) == len(values), "the set and dict should be equal size"
    total = 0
    for i in set1:
        total = total + values.get(str(i))
    return total

def shuffle_features(X):
    X_trans = list(map(list, zip(*X)))
    shuffle(X_trans)
    return list(map(list, zip(*X_trans)))

def remove_features(X,amount):
    X2 = []
    assert amount < len(X[0])
    for i in range(0,len(X)):
        X2.append(X[i][:len(X[i])-amount])
    return X2

# removes *amount* features in dataset X based on feature importance generated from a RandomForestClassifier
# return dataset X with removed features and the position of the removed features
def remove_features_importance(X,y,amount):
    assert amount < len(X[0])
    XT = list(map(list, zip(*X)))
    clf = RandomForestClassifier()    
    clf.fit(X,y)
    featsImp = clf.feature_importances_
    toRemove = []
    for i in range(amount):
        index_min = min(range(len(featsImp)), key=featsImp.__getitem__)
        toRemove.append(index_min)
        featsImp[index_min] = 1.1
    XT2 = []
    for i,xtj in enumerate(XT):
        if not i in toRemove:
            XT2.append(xtj)
    return list(map(list, zip(*XT2))),toRemove


def remove_features_MutualInformation(X,y,cat,amount):
    assert amount < len(X[0])
    XT = list(map(list, zip(*X)))
    clf = RandomForestClassifier()    
    clf.fit(X,y)
    featsImp = [adjusted_mutual_info_score(np.array(y), column[0]) for column in np.array(X)[:, np.where(cat)].T]
    toRemove = []
    for i in range(amount):
        index_min = min(range(len(featsImp)), key=featsImp.__getitem__)
        toRemove.append(index_min)
        featsImp[index_min] = 1.1
    XT2 = []
    for i,xtj in enumerate(XT):
        if not i in toRemove:
            XT2.append(xtj)
    return list(map(list, zip(*XT2))),toRemove

# removes *amount* features in dataset X based on correlation
    # return dataset X with removed features and the position of the removed features
def remove_features_correlationMax(X,y,amount):
    assert amount < len(X[0])
    toRemove = []
    XT = list(map(list, zip(*X)))
    correlation = np.corrcoef(XT,rowvar = True)
    for k in range(amount):        
        allValues = []
        for i in correlation:
            for j in i:
                allValues.append(abs(j))
        num = max(range(len(allValues)), key=allValues.__getitem__)
        row = int((num - (num % len(correlation[0])))/len(correlation[0]))
        collumn = num % len(correlation[0])
        if sum(correlation[row]) > sum(correlation[collumn]):
            toRemove.append(row)            
        else:
            toRemove.append(collumn)
        for j,item in enumerate(correlation):
            correlation[j][toRemove[k]] = 0
        for j,item in enumerate(correlation[toRemove[k]]):
            correlation[toRemove[k]][j] = 0
    XT2 = []   
    for i,xtj in enumerate(XT):
        if not i in toRemove:
            XT2.append(xtj)   
    
    return list(map(list, zip(*XT2))),toRemove


# removes features based on feature importance done on the first entry in dataset XList and target y
def remove_features_importance2(XList,y,amount):
    assert amount < len(XList[0][0])
    assert len(XList[0]) == len(y)
    for i in XList:
        assert len(i[0]) == len(XList[0][0])
#    XT = list(map(list, zip(*XList[0])))
    XTList = []
    XT2 = []
    for i in XList:
        XT2.append([])
        XTList.append(list(map(list, zip(*i))))
    clf = RandomForestClassifier()    
    clf.fit(XList[0],y)
    featsImp = clf.feature_importances_
    toRemove = []
    for i in range(amount):
        index_min = min(range(len(featsImp)), key=featsImp.__getitem__)
        toRemove.append(index_min)
        featsImp[index_min] = 1.1
    for i,xtj in enumerate(XTList[0]):
        if not i in toRemove:
            for j,item in enumerate(XT2):
                XT2[j].append(XTList[i])
    XList2 = []
    for i in XT2:
        XList2.append(list(map(list, zip(*XT2))))
        
    return list(map(list, zip(*XT2))),toRemove


# removes the features of dataset X based on the identifier in the list idens:
# returns dataset X with the removed features
def remove_featuresIdens(X,idens):
    XT = list(map(list, zip(*X)))
    XT2 = []
    for i,item in enumerate(XT):
        if not i in idens:
            XT2.append(XT[i])
    return list(map(list, zip(*XT2)))

# reduces the size of the datasets X and y by amount 
# returns datasets X and y reduced by size len(y)-amount
def split(X,y,amount):
    X = X[:amount]
    y = y[:amount]
    return X,y

def extendD(X,y,amount):
    if amount > 1.0:
        extender = amount *len(X)
    X2 = copy(X)
    y2 = copy(y)
    lenXy = len(X)
    for i in range(extender):
        X2.append(X[i%lenXy])
        y2.append(y[i%lenXy])
    return X2,y2

# the order of the dataset X is appended to the dataset X
def add_identifiers(X):
    for i,x in enumerate(X):
        X[i].append(i)
    return X

# the last collumn of the dataset X is removed 
# returns the dataset X with the last collumn removed
def split_identifiers(X):
    iden = []
    for i,x in enumerate(X):
        iden.append(X[i].pop())
    return X,iden
        
def orderX(X,y,iden):
    shuX = copy(X)
    shuy = copy(y)
    for i in range(0,len(shuy)):
        shuX[i].append(shuy[i])
    newX = []
    for i,x in enumerate(iden):
        newX.append(shuX[x])
    shuy = []
    for i in range(0,len(shuX)):
        shuy.append(shuX[i].pop())
    return newX,shuy     
        
def reduce_dataset(X,y,amount):
    X,y = shuffle_set(X,y)
    
    return X[:len(X)-round(amount*len(X))],y[:len(y)-round(amount*len(y))]

def remove_features2(X,amount):
    X2 = list(map(list, zip(*X)))
    for i in range(amount):
        if len(X2) == 1:
            print("Error removing full dataset")
            return list(map(list, zip(*X2)))
        X2.pop()    
    return list(map(list, zip(*X2)))


def remove_features2Cat(X,cat,amount):
    cat2 = copy(cat)
    X2 = list(map(list, zip(*X)))
    for i in range(amount):
        if len(X2) == 1:
            print("Error removing full dataset")
            return list(map(list, zip(*X2)))
        X2.pop()
        cat2.pop()
    return list(map(list, zip(*X2))),cat2
    
def create_features(y):
    X = []
    for i in y:
        X.append([i,i])
    return X

def features(did,amount):
    list1 = []
    list1.append(np.log(1-amount))
    list1.append(np.log((1-amount)*readDict(did)['NumberOfInstances']))
    list1.append(readDict(did)['NumberOfFeatures'])
    list1.append(readDict(did)['NumberOfInstances'])
    list1.append(readDict(did)['NumberOfSymbolicFeatures'])
    list1.append(readDict(did)['NumberOfNumericFeatures'])
    list1.append(did)
    list1.append(readDict(did)['NumberOfClasses'])
    return list1

def add_noise_featuresN(X,cat,amount,n):
#    assert amount >= 1, "features should be added"
    assert type(amount) == int , "amount should be integer"
    temp = deepcopy(X)
    cats = cat_needed(X,cat,amount)               
    for i in temp:
        for j in range(0,amount):
            if cats > 0:
                i.append(int(n*random.random()))
                cats = cats - 1
            else:
                i.append(n*random.random())
        cats = cat_needed(X,cat,amount)
    return temp

def add_noise_featuresN2(X,cat,amount,n):
#    assert amount >= 1, "features should be added"
    assert type(amount) == int , "amount should be integer"
    temp = deepcopy(X)
    cats = cat_needed(X,cat,amount)               
    for i in temp:
        for j in range(0,amount):
            if cats > 0:
                i.append(int(n*random.random()))
                cats = cats - 1
            else:
                i.append(n*random.random())
        cats = cat_needed(X,cat,amount)
    return temp

def durationPair(list1,start):
    return list1[start] + list1[start+1]

def PreSteps(clfName):
    if (clfName == 'SGDClassifier'):
        cats = [True,'both']
        steps = [OneHotEncoder(sparse = False, handle_unknown='ignore'),StandardScaler()]
    if (clfName == 'KNeighborsClassifier'):
        cats = [True,'both']
        steps = [OneHotEncoder(sparse = False, handle_unknown='ignore'),StandardScaler()]
    elif (clfName == '1NeighborsClassifier'):
        cats = [True,'both']
        steps = [OneHotEncoder(sparse = False, handle_unknown='ignore'),StandardScaler()]  
    
    elif (clfName[:4] == 'SVC-'):
        cats = [True,'both']
        steps = [OneHotEncoder(sparse = False, handle_unknown='ignore'),StandardScaler()]
    elif (clfName == 'MLPClassifier'):        
        cats = [True,'both']
        steps = [OneHotEncoder(sparse = False, handle_unknown='ignore'),StandardScaler()]
    
    return steps,cats

def preProcess(X_train,train_X,X_test,test_XIn,cat,clfName):
    
    steps,cats = PreSteps(clfName)
    try:
        len(steps)
    except NameError:
        print('no preprocessing steps found')
        
    XC_train,XN_train,XC_test,XN_test = splitCat(X_train,X_test,cat)
    for i,step in enumerate(steps):
        XC_train,XC_test,XN_train,XN_test = process(cats[i],XC_train,XC_test,XN_train,XN_test,step)        
    steps,cats = PreSteps(clfName)
    X_train,X_test = combine(list(XC_train),list(XN_train),list(XC_test),list(XN_test))
    cat = balance(cat,train_X)
    train_XC,train_XN,test_XC,test_XN = splitCat(train_X,test_XIn,cat)
    for i,step in enumerate(steps):
        train_XC,test_XC,train_XN,test_XN = process(cats[i],train_XC,test_XC,train_XN,test_XN,step)
        
    train_X,test_X = combine(list(train_XC),list(train_XN),list(test_XC),list(test_XN))
    
    return X_train,train_X,X_test,test_X


def splitCat(X_train2,X_test2,cat):
    assert len(X_train2[0]) == len(cat), "There should be equal categories as sample features in train"
    assert len(X_train2[0]) == len(X_test2[0]), "splits should be equal length"
    assert len(X_test2[0]) == len(cat), "There should be equal categories as sample features in test"
    Xt_train2 = list(map(list, zip(*X_train2)))
    Xt_test2 = list(map(list, zip(*X_test2)))
    XtC_train = []
    XtN_train = []
    XtC_test = []
    XtN_test = []
    for i,boole in enumerate(cat):
        if boole:
            if i >= len(Xt_test2):
                print('wut')
            XtC_train.append(Xt_train2[i])
            XtC_test.append(Xt_test2[i])
        else:
            XtN_train.append(Xt_train2[i])
            XtN_test.append(Xt_test2[i])
    XN_train = list(map(list, zip(*XtN_train)))
    XN_test = list(map(list, zip(*XtN_test)))
    XC_train = list(map(list, zip(*XtC_train)))
    XC_test = list(map(list, zip(*XtC_test)))
    return XC_train,XN_train,XC_test,XN_test

def balance(cat,X):
    catCopy = copy(cat)
    if len(cat) < len(X[0]):        
        lenC = len(cat)
        for i in range(lenC,len(X[0])):
            if round(X[0][i]) == X[0][i]:
                catCopy.append(True)
            else:
                catCopy.append(False)
    elif len(cat) > len(X[0]):
        lenC = len(cat)
        for i in range(len(X[0]),lenC):
            catCopy.pop()
    
    return catCopy

def combine(XC_train,XN_train,XC_test,XN_test):
    XtN_train = list(map(list, zip(*XN_train)))
    XtN_test = list(map(list, zip(*XN_test)))
    XtC_train = list(map(list, zip(*XC_train)))
    XtC_test = list(map(list, zip(*XC_test)))
    Xt_train2 = []
    Xt_test2 = []
    while len(XtN_train) > 0  and  len(XtN_test) > 0:        
        Xt_train2.append(XtN_train.pop(0))
        Xt_test2.append(XtN_test.pop(0))
    while len(XtC_test) > 0 and  len(XtC_train) > 0:
        Xt_train2.append(XtC_train.pop(0))
        Xt_test2.append(XtC_test.pop(0))
    X_train2 = list(map(list, zip(*Xt_train2)))
    X_test2 = list(map(list, zip(*Xt_test2)))
    return X_train2,X_test2

def process(processOverType,XC_train,XC_test,XN_train,XN_test,step):
    if processOverType == 'both':
        if len(XC_train) > 0:
            step.fit(XC_train)
            XC_train = step.transform(XC_train)
            XC_test = step.transform(XC_test)
        if len(XN_train) > 0:
            step.fit(XN_train)
            XN_train = step.transform(XN_train)
            XN_test = step.transform(XN_test)
    elif processOverType == True:
        if len(XC_train) > 0:
            step.fit(XC_train)
            XC_train = step.transform(XC_train)
            XC_test = step.transform(XC_test)
    else:
        if len(XN_train) > 0:
            step.fit(XN_train)
            XN_train = step.transform(XN_train)
            XN_test = step.transform(XN_test)
    return XC_train,XC_test,XN_train,XN_test


class OneHotEncoderSelf(object):
    def __init__(self):
        OneHotEncoder()
        
    def fit(X):
        return OneHotEncoder.fit(X)
    def transform(X):
        transformed = OneHotEncoder.transform(X)
        return transformed.toarray()

def preProcessV2(X_train,train_X,X_test,test_X,cat,catM,clfName):    
    steps,cats = PreSteps(clfName)
    try:
        len(steps)
    except NameError:
        print('no preprocessing steps found')        
    XC_train,XN_train,XC_test,XN_test = splitCat(X_train,X_test,cat)
    for i,step in enumerate(steps):
        XC_train,XC_test,XN_train,XN_test = process(cats[i],XC_train,XC_test,XN_train,XN_test,step)        
    steps,cats = PreSteps(clfName)
    X_train,X_test = combine(list(XC_train),list(XN_train),list(XC_test),list(XN_test))
    train_XC,train_XN,test_XC,test_XN = splitCat(train_X,test_X,catM)
    for i,step in enumerate(steps):
        train_XC,test_XC,train_XN,test_XN = process(cats[i],train_XC,test_XC,train_XN,test_XN,step)
        
    train_X,test_X = combine(list(train_XC),list(train_XN),list(test_XC),list(test_XN))
    
    return X_train,train_X,X_test,test_X

def random_seed(seed):
    random.seed(seed)
    return random.random()

#returns the 1st and quartiles 
def CIIntervals(ScoreLine):
    lines = [[],[],[],[]]
    for i in ScoreLine:
        val = CIScores(i)
        for j,item in enumerate(val):
            lines[j].append(item)
    return lines


def CIScores(ResultList):
    inp = sorted(ResultList)
    Cis = []
    Cis.append(inp[round(len(inp)/4)])
    Cis.append(inp[round(len(inp)/2)])
    Cis.append(inp[round(len(inp)/1.5)])
    Cis.append(sum(inp)/len(inp))
    return Cis


    

            
                