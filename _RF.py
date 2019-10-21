import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from datetime import datetime
from sklearn.model_selection import train_test_split 
data=pd.read_csv('wine.csv')
bins = [1,4,6,10]
quality_labels=[0,1,2]
data['quality_categorical'] = pd.cut(data['quality'], bins=bins, labels=quality_labels, include_lowest=True)
#display(data.head(n=2))#print(data.tail(n=12zz))
quality_raw = data['quality_categorical']
features_raw = data.drop(['quality', 'quality_categorical'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(features_raw,  quality_raw,  test_size = 0.2,  random_state = 0)
# Show the results of the split: print("Training set has {} samples.".format(X_train.shape[0])) print("Testing set has {} samples.".format(X_test.shape[0]))

from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score

def train_predict_evaluate(learner, sample_size,X_train,y_train,X_test,y_test):
    results={}
    start=datetime.now().time()
    learner=learner.fit(X_train[:sample_size],y_train[:sample_size])
    end=datetime.now().time()

   # results['training_time']=end-start

    #prediction of the first 3000
    prediction_train=learner.predict(X_train[:300])
    prediction_test=learner.predict(X_test)

    #end=datetime.now().time()

    #results['pred_time']=end-start
    results['acc_train']=accuracy_score(y_train[:300], prediction_train)
    results['acc_test']=accuracy_score(y_test, prediction_test)
    results['f_train']=fbeta_score(y_train[:300], prediction_train,beta=0.5, average='micro')
    results['f_test']=fbeta_score(y_test, prediction_test,beta=0.5, average='micro')
    print("{} trained on {} samples.".format(learner.__class__.__name__,sample_size))

    return results

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

clf_C = RandomForestClassifier(max_depth=None, random_state=None)

# Calculate the number of samples for 1%, 10%, and 100% of the training data
# HINT: samples_100 is the entire training set i.e. len(y_train)
# HINT: samples_10 is 10% of samples_100
# HINT: samples_1 is 1% of samples_100

samples_100 = len(y_train)
samples_10 = int(len(y_train)*10/100)
samples_1 = int(len(y_train)*1/100)

# Collect results on the learners
results = {}
for clf in [ clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] =  train_predict_evaluate(clf, samples, X_train, y_train, X_test, y_test)

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
clf=RandomForestClassifier(max_depth=None, random_state=None)
parameters={'n_estimators':[10,20,30], 'max_features':[3,4,5,None], 'max_depth':[5,6,7,None]}
scorer=make_scorer(fbeta_score, beta=0.5, average="micro")
grid_obj=GridSearchCV(clf,parameters,scoring=scorer)
grid_fit=grid_obj.fit(X_train, y_train)
best_clf=grid_fit.best_estimator_
predictions=(clf.fit(X_train,y_train)).predict(X_test)
best_preditions=best_clf.predict(X_test)

# print("unoptimized mode\n...........")
# print("Accuracy score on testing data:{:.4f}".format(accuracy_score(y_test,predictions)))
# print("F-score on testing data: {:4f}".format(fbeta_score(y_test,predictions,beta=0.5,average="micro")))
# print(best_clf)
# print("Optimized mode\n...........")
# print("\nFinal Accuracy score on testing data:{:.4f}".format(accuracy_score(y_test,best_preditions)))
# print("Final F-score on testing data: {:.4f}".format(fbeta_score(y_test,best_preditions,beta=0.5,average="micro")))

wine_data=pd.read_csv('wine_test.csv')
for i, quality in enumerate(best_clf.predict(wine_data)):
    print("predicted quality for Wine {} is: {}".format(i+1,quality))
