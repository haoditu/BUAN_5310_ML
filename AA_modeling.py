import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection  import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import os

#os.chdir("/Users/Shenshen_Wu/Documents/GitHub/BUAN_5310_ML")


acs_airport = pd.read_csv("acs_ap.csv", sep=',', index_col=0)
acs_airline = pd.read_csv("acs_al.csv", sep=',', index_col=0)

# Summary statistics
pd.set_option('display.max_columns', None)
acs_airport.describe()
acs_airline.describe()


# Correlation
acs_airport.corr()
acs_airline.corr()

### Airport modeling
# Covert categorical variables to numerical
acs_airport['Airport'] = acs_airport['Airport'].astype('category').cat.codes
acs_airport['Airline'] = acs_airport['Airline'].astype('category').cat.codes
#acs_airport['TripPurpose'] = acs_airport['TripPurpose'].astype('category').cat.codes
acs_airport['Age'] = acs_airport['Age'].astype('category').cat.codes
acs_airport['Income'] = acs_airport['Income'].astype('category').cat.codes
acs_airport['ModeTransport'] = acs_airport['ModeTransport'].astype('category').cat.codes
acs_airport['Occupation'] = acs_airport['Occupation'].astype('category').cat.codes
acs_airport['Occupation'] = acs_airport['Occupation'].astype('category').cat.codes
acs_airport['Nationality'] = acs_airport['Nationality'].astype('category').cat.codes


Y_ap= acs_airport['Airport']
X_ap= acs_airport.drop(['Airport'],axis = 1)

# Logit model 
logit_model = sm.Logit(Y_ap, X_ap).fit()
logit_model.summary()
X_train, X_test, Y_train, Y_test = train_test_split(X_ap, Y_ap, test_size=0.30,random_state=109)
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, Y_train)
Y_pred = logisticRegr.predict(X_test)
print(metrics.confusion_matrix(Y_test, Y_pred))  
print(classification_report(Y_test, Y_pred))
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Error Rate:",1-metrics.accuracy_score(Y_test, Y_pred))

# Recursive Feature Elimination
# Reference: https://www.datacamp.com/community/tutorials/feature-selection-python
#from sklearn.feature_selection import RFE
#from sklearn.linear_model import LogisticRegression

# Feature ranking and extraction
model = LogisticRegression()
rfe = RFE(model, 5)
fit = rfe.fit(X_ap, Y_ap)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))  
# Top 5 features: Airline,Age, Destination, DepartureTime, Income


### Tree Model
X_train, X_test, Y_train, Y_test = train_test_split(X_ap, Y_ap, test_size=0.30,random_state=109)  
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10, max_features=None, max_leaf_nodes=None, min_samples_leaf=10, min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False, random_state=100, splitter='best')
clf = clf.fit(X_train, Y_train)
# export estimated tree into dot graphic file
dot_data = tree.export_graphviz(clf, out_file='Dtree_airport.dot', feature_names=X_ap.columns)
Y_pred = clf.predict(X_test)  
print(metrics.confusion_matrix(Y_test, Y_pred))  
print(classification_report(Y_test, Y_pred))
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Error Rate:",1-metrics.accuracy_score(Y_test, Y_pred))

# change tree parameters
X_train, X_test, Y_train, Y_test = train_test_split(X_ap, Y_ap, test_size=0.30,random_state=109)  
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5, max_features=None, max_leaf_nodes=None, min_samples_leaf=3, min_samples_split=3, min_weight_fraction_leaf=0.0, presort=False, random_state=100, splitter='best')
clf = clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)  
print(metrics.confusion_matrix(Y_test, Y_pred)) 
print(classification_report(Y_test, Y_pred)) 
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Error Rate:",1-metrics.accuracy_score(Y_test, Y_pred))


### SVM model
X_train, X_test, y_train, y_test = train_test_split(X_ap, Y_ap, test_size=0.30,random_state=109)  

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


### Airline modeling
# Covert categorical variables to numerical
acs_airline['Airport'] = acs_airport['Airport'].astype('category').cat.codes
acs_airline['Airline'] = acs_airport['Airline'].astype('category').cat.codes
acs_airline['TripPurpose'] = acs_airport['TripPurpose'].astype('category').cat.codes
acs_airline['Age'] = acs_airport['Age'].astype('category').cat.codes
acs_airline['Income'] = acs_airport['Income'].astype('category').cat.codes
Y_al= acs_airport['Airline']
X_al= acs_airport.drop(['Airline', 'ProvinceResidence'], axis = 1)

# Logit model 
logit_model = sm.Logit(Y_al, X_al).fit()
logit_model.summary()

X_train, X_test, Y_train, Y_test = train_test_split(X_al, Y_al, test_size=0.30,random_state=109)
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, Y_train)
Y_pred = logisticRegr.predict(X_test)
print(metrics.confusion_matrix(Y_test, Y_pred))  
print(classification_report(Y_test, Y_pred))
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Error Rate:",1-metrics.accuracy_score(Y_test, Y_pred))

# Feature ranking and extraction
model = LogisticRegression()
rfe = RFE(model, 5)
fit = rfe.fit(X_al, Y_al)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))  
# Top 5 features: Airport,NoTripLastYear,DepartureTime,FlyingCompany,GroupTravel


### Tree Model
X_train, X_test, Y_train, Y_test = train_test_split(X_al, Y_al, test_size=0.30,random_state=109)  
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10, max_features=None, max_leaf_nodes=None, min_samples_leaf=10, min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False, random_state=100, splitter='best')
clf = clf.fit(X_train, Y_train)
# export estimated tree into dot graphic file
dot_data = tree.export_graphviz(clf, out_file='Dtree_airline1.dot', feature_names=X_al.columns)
Y_pred = clf.predict(X_test)  
print(metrics.confusion_matrix(Y_test, Y_pred))  
print(classification_report(Y_test, Y_pred))
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Error Rate:",1-metrics.accuracy_score(Y_test, Y_pred))

# change tree parameters
X_train, X_test, Y_train, Y_test = train_test_split(X_al, Y_al, test_size=0.30,random_state=109)  
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5, max_features=None, max_leaf_nodes=None, min_samples_leaf=3, min_samples_split=3, min_weight_fraction_leaf=0.0, presort=False, random_state=100, splitter='best')
clf = clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)  
print(metrics.confusion_matrix(Y_test, Y_pred)) 
print(classification_report(Y_test, Y_pred)) 
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Error Rate:",1-metrics.accuracy_score(Y_test, Y_pred))


### SVM model
X_train, X_test, y_train, y_test = train_test_split(X_al, Y_al, test_size=0.30,random_state=109)  

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

