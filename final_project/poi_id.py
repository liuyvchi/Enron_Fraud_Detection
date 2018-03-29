#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data,test_classifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile,SelectKBest
from sklearn.preprocessing import MinMaxScaler

# Load data
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# explore the dataset
print "the total number of records",len(data_dict)

for record in data_dict.values():
    print "the total number of features", len(record)-1
    print "the names of features:", record    
    break

poi_records = []

for record in data_dict.items():    
    if record[1]['poi'] == True:
        poi_records.append(record)
    
print "the total number of poi:", len(poi_records)

# look through the dataset to see the 'NaN' conditions for different features

features_list = ['poi','salary','bonus','total_payments','deferral_payments','exercised_stock_options',\
                 'restricted_stock','restricted_stock_deferred','total_stock_value','expenses',\
                 'other','director_fees','loan_advances','deferred_income','long_term_incentive',\
                 'from_poi_to_this_person','from_this_person_to_poi','to_messages','from_messages',\
                 'shared_receipt_with_poi']

print "NaN numbers for each feature:"

for feature in features_list:
    nan_num = 0
    for record in data_dict.items():    
        for key, value in record[1].items():
            if feature == key and value == "NaN":
                nan_num += 1
    print feature, nan_num, "\n"

# Find outliers in the data
for record in data_dict.items():
    plt.scatter(record[1]["salary"], record[1]["bonus"])
    
plt.xlabel("bonus")
plt.ylabel("salary")
plt.show()

#there are records where salary and bonus are extreamlly high. we need find these records
salary_outliers = []
bonus_outliers = []

for record in data_dict.items():
    salary = record[1]['salary']
    if salary == "NaN":
        continue
    salary_outliers.append(salary)
    
for record in data_dict.items():       
    bonus = record[1]['bonus']
    if bonus == "NaN":
        continue
    bonus_outliers.append(bonus)

#sort to find the max 3 ones in each list
salary_outliers.sort(reverse = True)
bonus_outliers.sort(reverse = True)

print salary_outliers[:3]
print bonus_outliers[:3]

#we can find that the max one in each list is significantlly higher than others. 
# We assume that the record which contains above max values is outlier.
for record in data_dict.items():
    if record[1]['salary'] == salary_outliers[0]:
        print record[0]
        break
        
for record in data_dict.items():
    if record[1]['bonus'] == bonus_outliers[0]:
        print record[0]
        break

#For some records with too much NaN features, we can assum they may are not hunman related records and need to be removed. 
# Also, if all features in a record are NaN, it is a unvaluble record and should be remove.
for record in data_dict.items():
    NaN_num = 0
    for feature in record[1].items():
        if feature[1] == "NaN":
            NaN_num += 1
    if 17 < NaN_num < 20:
        print record[0], ", NaN_num:", NaN_num  
    if NaN_num == 20:
        print record[0], ", NaN_num:", NaN_num  
# task2 Remove outliners
data_dict.pop("TOTAL")
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")
data_dict.pop("LOCKHART EUGENE E")


### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features
# Task 3: Create new feature(s)
# we assum that suspects contact with poi frequetly. 
# Therefore we introduce the features "from_poi_ratio" and "to_poi_ratio".

for key in data_dict:
    if data_dict[key]['from_poi_to_this_person'] == "NaN" or data_dict[key]['to_messages'] == "NaN":
        data_dict[key]['to_ratio'] = "NaN"
    else:
        data_dict[key]['to_ratio'] = float(data_dict[key]['from_poi_to_this_person']/data_dict[key]['to_messages'])     
        
    if data_dict[key]['from_this_person_to_poi'] == "NaN" or data_dict[key]['from_messages'] == "NaN":
        data_dict[key]['from_ratio'] = "NaN"
    else:
        data_dict[key]['from_ratio'] = float(data_dict[key]['from_this_person_to_poi']/data_dict[key]['from_messages']) 


features_list = ['poi','salary','bonus','total_payments','deferral_payments','exercised_stock_options',\
                 'restricted_stock','restricted_stock_deferred','total_stock_value','expenses',\
                 'other','director_fees','loan_advances','deferred_income','long_term_incentive',\
                 'from_poi_to_this_person','from_this_person_to_poi','to_messages','from_messages',\
                 'shared_receipt_with_poi','to_ratio','from_ratio']

my_dataset = data_dict

### Task 1: Select what features you'll use.

# Feature selection and scalling
# use SelectKBest algorithm to select best 10 features
def Select_K_Best(data_dict, features_list, k):
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)
    k_best_modal = SelectKBest(k=k)
    k_best_modal.fit(features, labels)
    scores = k_best_modal.scores_
    tuples_unsorted = zip(features_list[1:], scores)
    k_best_features = sorted(tuples_unsorted, key=lambda x: x[1], reverse=True)   
    return k_best_features[:k]

selected_features = Select_K_Best(my_dataset,features_list,5)
selected_features_4 = Select_K_Best(my_dataset,features_list,4)
selected_features_6 = Select_K_Best(my_dataset,features_list,6)
print "selected_features:", selected_features
print "selected_features_4:", selected_features_4
print "selected_features_6:", selected_features_6

my_features_list = ['poi'] 
my_features_list_4 = ['poi'] 
my_features_list_6 = ['poi'] 
for feature in selected_features:
    my_features_list = my_features_list + [feature[0]]
for feature in selected_features_4:
    my_features_list_4 = my_features_list_4 + [feature[0]]
for feature in selected_features_6:
    my_features_list_6 = my_features_list_6 + [feature[0]]
    
print "my_features_list:",my_features_list

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_features_list, sort_keys = True)
data_4 = featureFormat(my_dataset, my_features_list_4, sort_keys = True)
data_6 = featureFormat(my_dataset, my_features_list_6, sort_keys = True)
labels, features = targetFeatureSplit(data)
labels_4, features_4 = targetFeatureSplit(data_4)
labels_6, features_6 = targetFeatureSplit(data_6)


# we normalize data by feature scalling
scaler = MinMaxScaler()
features_new = scaler.fit_transform(features)
features_new_4 = scaler.fit_transform(features_4)
features_new_6 = scaler.fit_transform(features_6)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# divide data into training data and test data
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features_new, labels, test_size=0.3, random_state=42)
features_train_4, features_test_4, labels_train_4, labels_test_4 = \
    train_test_split(features_new_4, labels_4, test_size=0.3, random_state=42)
features_train_6, features_test_6, labels_train_6, labels_test_6 = \
    train_test_split(features_new_6, labels_6, test_size=0.3, random_state=42)
    
from sklearn.metrics import precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from time import time
from sklearn.model_selection import GridSearchCV

# Provided to give you a starting point. Try a variety of classifiers.

#GaussianNB
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

print "\n","select top 5 features"
begin = time()

clf_NB = GaussianNB()
parm = {}
clf_NB = Pipeline([('scaler',scaler),('gnb',clf_NB)])
gs = GridSearchCV(clf_NB, parm)
gs.fit(features_train,labels_train)
clf_NB = gs.best_estimator_
pred = clf_NB.predict(features_test)

end = time()

accuracy = accuracy_score(labels_test,pred)

print "GaussianNB: "
print clf_NB.score(features_train,labels_train)
print test_classifier(clf_NB,my_dataset,my_features_list)
print "time consuming: ", end - begin

# to see other features numbers performance on GaussianNB. to see how many features are best for modalling

print "\n","select top 4 features"
begin = time()

clf_NB = GaussianNB()
parm = {}
clf_NB = Pipeline([('scaler',scaler),('gnb',clf_NB)])
gs = GridSearchCV(clf_NB, parm)
gs.fit(features_train_4,labels_train_4)
clf_NB = gs.best_estimator_
pred = clf_NB.predict(features_test_4)

end = time()

accuracy = accuracy_score(labels_test_4,pred)

print accuracy
print test_classifier(clf_NB,my_dataset,my_features_list_4)
print "time consuming: ", end - begin

print "\n","select top 6 features"
begin = time()

clf_NB = GaussianNB()
parm = {}
clf_NB = Pipeline([('scaler',scaler),('gnb',clf_NB)])
gs = GridSearchCV(clf_NB, parm)
gs.fit(features_train_6,labels_train_6)
clf_NB = gs.best_estimator_
pred = clf_NB.predict(features_test_6)

end = time()

accuracy = accuracy_score(labels_test_6,pred)

print accuracy
print test_classifier(clf_NB,my_dataset,my_features_list_6)
print "time consuming: ", end - begin
## we finally choose 5 features.


# SVM
print "\n"
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

begin = time()

clf_SVC = SVC()
parms = {'svc__kernel':('linear','rbf'),'svc__C':[1.0,2.0]}
pipeline = Pipeline([('scaler',scaler),('svc',clf_SVC)])
gs = GridSearchCV(pipeline, parms)
gs.fit(features_train,labels_train)
clf_SVC = gs.best_estimator_
pred = clf_SVC.predict(features_test)

end = time()

print "SVM: "
print clf_SVC.score(features_train,labels_train)
print test_classifier(clf_SVC,my_dataset,my_features_list)
print "time consuming: ", end - begin

# RandomForest
print "\n"
from sklearn.ensemble import RandomForestClassifier

begin = time()

clf_RF = RandomForestClassifier()
parms = {'criterion': ['gini', 'entropy'], \
         'max_depth': [None, 3, 5, 10], \
         'max_leaf_nodes': [None, 5, 10, 20], \
         'n_estimators': [1, 5, 10, 50, 100]}
gs = GridSearchCV(clf_RF, parms)
gs.fit(features_train,labels_train)
clf_RF = gs.best_estimator_
pred = clf_RF.predict(features_test)

end = time()

print "RandomForest:"
print clf_RF.score(features_train,labels_train)
print test_classifier(clf_RF,my_dataset,my_features_list)
print "time consuming: ", end - begin

# AdaBoost
print "\n"
from sklearn.ensemble import AdaBoostClassifier

begin = time()

clf_AB = AdaBoostClassifier()
parms = {'learning_rate': [0.05, 0.1, 0.5, 2.0], \
              'algorithm': ['SAMME', 'SAMME.R'], \
              'n_estimators': [1, 5, 10, 50, 100]}
gs = GridSearchCV(clf_AB, parms)
gs.fit(features_train,labels_train)
clf_AB = gs.best_estimator_
pred = clf_AB.predict(features_test)

end = time()

print "AdaBoost:"
print clf_AB.score(features_train,labels_train)
print test_classifier(clf_AB,my_dataset,my_features_list)
print "time consuming: ", end - begin


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
clf = clf_NB

dump_classifier_and_data(clf, my_dataset, my_features_list)
