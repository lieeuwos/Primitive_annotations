# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:27:44 2017

@author: S127788
"""
import random
from copy import copy
from copy import deepcopy
from random import shuffle,random
import numpy as np

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
    assert amount >= 1, "features should be added"
    assert type(amount) == int , "amount should be integer"
    temp = deepcopy(X)
    for i in temp:
        for j in range(0,amount):
            i.append(random())            
    return temp

def add_noise_features2(X,cat,amount):
    assert amount >= 1, "features should be added"
    assert type(amount) == int , "amount should be integer"
    temp = deepcopy(X)
    cats = cat_needed(X,cat,amount)
    for i in temp:
        if cats > 0:
            for j in range(0,amount):
                i.append(int(3*random()))
            cats = cats - 1
        else: 
            for j in range(0,amount):
                i.append(random())            
    return temp

def cat_needed(X,cat,amount):
    count = 0
    for i in cat:
        if i:
            count = count + 1
    return count/len(cat)*amount
    

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



    