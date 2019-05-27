import os
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection  import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 



os.chdir("/Users/Shenshen_Wu/Documents/GitHub/BUAN_5310_ML")

acs = pd.read_csv("acs.csv", sep=',', header=0)

# Summary statistics
pd.set_option('display.max_columns', None)
acs.describe()

# Correlation
acs.corr()

Y = acs['Airport'].astype('category').cat.codes
X = acs.drop(['Airport'],axis = 1)

### Logit model 
logit_model = sm.Logit(Y, X).fit()
logit_model.summary()

# Recursive Feature Elimination
# Reference: https://www.datacamp.com/community/tutorials/feature-selection-python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Feature extraction
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))  
# Top 3 features: Destination, DepartureTime, NoTransport


### Tree Model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30,random_state=109)  
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10, max_features=None, max_leaf_nodes=None, min_samples_leaf=10, min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False, random_state=100, splitter='best')
clf = clf.fit(X_train, Y_train)
# export estimated tree into dot graphic file
dot_data = tree.export_graphviz(clf, out_file='Dtree.dot', feature_names=X.columns)
Y_pred = clf.predict(X_test)  
print(metrics.confusion_matrix(Y_test, Y_pred))  
print(classification_report(Y_test, Y_pred))
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Error Rate:",1-metrics.accuracy_score(Y_test, Y_pred))

# change tree parameters
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30,random_state=109)  
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5, max_features=None, max_leaf_nodes=None, min_samples_leaf=3, min_samples_split=3, min_weight_fraction_leaf=0.0, presort=False, random_state=100, splitter='best')
clf = clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)  
print(metrics.confusion_matrix(Y_test, Y_pred)) 
print(classification_report(Y_test, Y_pred)) 
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Error Rate:",1-metrics.accuracy_score(Y_test, Y_pred))


### SVM model
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30,random_state=109)  

# Normalize data
scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 
svclassifier = SVC(kernel='linear')    	
svclassifier.fit(X_train, y_train)  
y_pred = svclassifier.predict(X_test)  	
print(metrics.confusion_matrix(y_test, y_pred)) 	
print(classification_report(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Test Erro Rate:",1-metrics.accuracy_score(y_test, y_pred))

