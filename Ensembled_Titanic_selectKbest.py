# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 18:44:47 2022

@author: Pixy_33

This entry has borrowed many ideas from the following:
https://www.kaggle.com/code/mnassrib/titanic-logistic-regression-with-python/notebook
https://amueller.github.io/aml/01-ml-workflow/04-categorical-variables.html
https://towardsdatascience.com/stop-wasting-useful-information-when-imputing-missing-values-d6ef91ef4c21
https://machinelearningmastery.com/feature-selection-with-categorical-data/
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


#Force pd to display print in full
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

train = pd.read_csv('train.csv')
#Random reshuffle of index to avoid possible implicit sorting
train = train.reindex(np.random.permutation(train.index))
test=pd.read_csv('test.csv')


#Check data
print('Data head=','\n',train.head(),'\n')
#See if data is imbalanced
print('Survived ratio=','\n',train['Survived'].value_counts()/train.shape[0],'\n')
#Check correlation to see features that are worth being selected
print('Correlation=','\n',train.corr(method='kendall'),'\n')
#Check the percentage of missing data
print('Missing data summary=','\n',train.isna().sum()/train.shape[0],'\n')
#Next we will use sns.pairplot() to do exploratory data analysis, but the plot
#can only handle a dataset with no missing data


#Age, Embarked and Cabin are missing data, but Cabin is missing >70% and the 
#location of passengers are likely accounted for in Pclass anyway, therefore
#Cabin will be left out
#Age will be imputed and since only 0.002 of Embarked are missing, so dropping 
#them is the easiest approach

#Dropping missing Embarked entries
train=train.dropna(subset=['Embarked'])

#Age is correlated with other features, i.e. SibSp, Parch, Fare therefore
#imputing with mean/median would have ignored this fact and may disrupt the original
#distribution, here IterativeImputer() will be used instead
#Imputing Age for train and test
numeric_features=['Age','SibSp','Parch','Fare']
data_for_imp=train[numeric_features]
imp=IterativeImputer()
data_for_imp[:]=imp.fit_transform(data_for_imp)
train['Age']=np.round(data_for_imp['Age'])

data_for_imp_test=test[numeric_features]
imp_test=IterativeImputer()
data_for_imp_test[:]=imp_test.fit_transform(data_for_imp_test)
test['Age']=np.round(data_for_imp_test['Age'])

#Now that missing datas have been imputed, we are ready to do exploratory data
#analysis using sns.pairplot()
cat_cols_pair = ['Pclass', 'Sex','Embarked']
cols_2_pair = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
for col in cat_cols_pair:
    sns.set()
    plt.figure()
    sns.pairplot(train[cols_2_pair], height=3.0, hue=col)
    plt.show()


#A few observations from the pairplot:
#3rd class passengers are much more likely to die (1st plot)
#Female are clearly more likely to have survived (2nd plot)
#For those who embarked in Southampton, their chances of survival are lower 
#than that of those who embarked elsewhere, as there are more deceased than
#survived Southampton passengers (3rd plot)
#One hot encoding Embarked, Sex and Pclass, the latter has values between 1-3 so 
#treating it as numeric feature may mislead the model to treat the values as weight
train=train.dropna(subset=['Embarked'])
ohe=OneHotEncoder(sparse=False)
ohe_data=train[['Pclass','Sex','Embarked']]
train_ohe=(ohe.fit_transform(ohe_data)).astype(int)
train_ohe = pd.DataFrame(train_ohe,columns=ohe.get_feature_names(ohe_data.columns),index=ohe_data.index)
train=pd.concat([train,train_ohe], axis=1)

#One hot encoding relevant columns in test
ohe_test=OneHotEncoder(sparse=False)
ohe_test_data=test[['Pclass','Sex','Embarked']]
test_ohe=(ohe_test.fit_transform(ohe_test_data)).astype(int)
test_ohe = pd.DataFrame(test_ohe,columns=ohe_test.get_feature_names(ohe_test_data.columns),index=ohe_test_data.index)
test=pd.concat([test,test_ohe], axis=1)


#It is suspected that children and teenagers were prioritised to board lifeboats
#A visualisation is created to see if that is worth investigating
fig, ax = plt.subplots()
sns.displot(train,x='Age',hue='Survived',multiple="stack", discrete=True)
plt.show()
#It seems that those who were under 16y/o were more likely to have survived
#Create a feature identifying passengers who are >16y/o
train['Over 16']=(train['Age']>16).astype(int)
test['Over 16']=(test['Age']>16).astype(int)


feature_cols = ['Over 16', 'Pclass_1','Pclass_2','Pclass_3','Sex_female','Embarked_C','Embarked_Q','Embarked_S']
#Features
X = train[feature_cols] 
#Target variable
y = train.Survived


#We do a final check on the relevance of these categorical features by doing a chi
#square test
fs = SelectKBest(score_func=chi2, k='all')
X_fs = fs.fit_transform(X, y)
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
#So it turns out that actually 'Over 16' is not a very relevant feature, while 
#features 1,3,4 and 5 are the most prominent

#Narrowing down selection of features based on chi square test
feature_cols = ['Pclass_1','Pclass_3','Sex_female','Embarked_C']
#Features
X = train[feature_cols] 


#In this project, logistic regression, random forest, and ensembling both will be
#tried and evaluated
#Initiating the models
logreg = LogisticRegressionCV()
rfc = RandomForestClassifier()
vc=VotingClassifier([('clf1',logreg),('clf2',rfc)],voting='soft')


logreg_accuracy = cross_val_score(logreg, X, y, cv=10, scoring='accuracy')
print(logreg.__class__.__name__+" average accuracy is",logreg_accuracy.mean())
#LogisticRegressionCV average accuracy is 0.793
rfc_accuracy = cross_val_score(rfc, X, y, cv=10, scoring='accuracy')
print(rfc.__class__.__name__+" average accuracy is",rfc_accuracy.mean())
#RandomForestClassifier average accuracy is 0.817
vc_accuracy = cross_val_score(vc, X, y, cv=10, scoring='accuracy')
print(vc.__class__.__name__+" average accuracy is" ,vc_accuracy.mean())
#VotingClassifier average accuracy is 0.818

#Therefore it seems that ensembling yields the highest accuracy
#Next we fine tune the hyperparameters to maximize performance
#Fitting the models to generate prediction
logreg.fit(X,y)
rfc.fit(X,y)
vc.fit(X,y)


#Using gridsearch to find the best set of parameters for random forest and ensemble
#Logistic regression has already been fine tuned with cross validation
rfc_params={'n_estimators': [100, 200, 300, 400, 500], 'max_features': ['auto', 'sqrt', 'log2'],
'max_depth' : [4,5,6,7,8]}
grid_rfc=GridSearchCV(rfc,rfc_params)
grid_rfc.fit(X,y)
print('Best params for RF=',grid_rfc.best_params_)
rfc_best_params=grid_rfc.best_params_

vc_params={'voting':['hard','soft'],'weights':[(1,1),(2,1),(1,2)]}
grid_vc=GridSearchCV(vc,vc_params)
grid_vc.fit(X,y)
print('Best params for VC=',grid_vc.best_params_)
vc_best_params=grid_vc.best_params_
#Best parameters are max_depth=4, max_features='auto', n_estimators=100

rfc_tuned=RandomForestClassifier(max_depth=rfc_best_params['max_depth'], 
                                 max_features=rfc_best_params['max_features'], 
                                 n_estimators=rfc_best_params['n_estimators'])
rfc_tuned.fit(X,y)
vc_tuned=VotingClassifier([('clf1',logreg),('clf2',rfc_tuned)],
                          voting=vc_best_params['voting'],
                          weights=vc_best_params['weights'])
vc_tuned.fit(X,y)


#Dropping columns so that column wise test_1 lines up with X
test_1=test[feature_cols]
test_pred=vc_tuned.predict(test_1)
test_1['PassengerId']=test['PassengerId']
test_1['Survived']=test_pred
test_1=test_1.drop(columns=feature_cols)
test_1.to_csv("Ensembled_Titanic_selectKbest.csv",index=False)

