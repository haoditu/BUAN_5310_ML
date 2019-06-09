import pandas as pd
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import os

os.chdir("/Users/Shenshen_Wu/Documents/GitHub/BUAN_5310_ML")



#### Airport Modeling
## data preprocessing
train_ap = pd.read_csv("train_airport.csv", sep=',', index_col=0)
test_ap = pd.read_csv("test_airport.csv", sep=',', index_col=0 )


X_train_ap = train_ap.drop(['Airport'],axis = 1)
Y_train_ap = train_ap['Airport'].astype('category').cat.codes

X_test_ap = test_ap.drop(['Airport'],axis = 1)
Y_test_ap = test_ap['Airport'].astype('category').cat.codes

X_train_ap["Airline"] = X_train_ap["Airline"].astype('category').cat.codes
X_train_ap["Nationality"] = X_train_ap["Nationality"].astype('category').cat.codes
X_train_ap["TripPurpose"] = X_train_ap["TripPurpose"].astype('category').cat.codes
X_train_ap["FrequentFlightDestination"] = X_train_ap["FrequentFlightDestination"].astype('category').cat.codes
X_train_ap["Destination"] = X_train_ap["Destination"].astype('category').cat.codes
X_train_ap["Occupation"] = X_train_ap["Occupation"].astype('category').cat.codes
X_train_ap["Income"] = X_train_ap["Income"].astype('category').cat.codes
X_train_ap["ProvinceResidence"] = X_train_ap["ProvinceResidence"].astype('category').cat.codes
X_train_ap["ModeTransport"] = X_train_ap["ModeTransport"].astype('category').cat.codes


X_test_ap["Airline"] = X_test_ap["Airline"].astype('category').cat.codes
X_test_ap["Nationality"] = X_test_ap["Nationality"].astype('category').cat.codes
X_test_ap["TripPurpose"] = X_test_ap["TripPurpose"].astype('category').cat.codes
X_test_ap["FrequentFlightDestination"] = X_test_ap["FrequentFlightDestination"].astype('category').cat.codes
X_test_ap["Destination"] = X_test_ap["Destination"].astype('category').cat.codes
X_test_ap["Occupation"] = X_test_ap["Occupation"].astype('category').cat.codes
X_test_ap["Income"] = X_test_ap["Income"].astype('category').cat.codes
X_test_ap["ProvinceResidence"] = X_test_ap["ProvinceResidence"].astype('category').cat.codes
X_test_ap["ModeTransport"] = X_test_ap["ModeTransport"].astype('category').cat.codes


### Tree Model 1
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10, max_features=None, max_leaf_nodes=None, min_samples_leaf=10, min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False, random_state=100, splitter='best')
clf = clf.fit(X_train_ap, Y_train_ap)

# export estimated tree into dot graphic file
dot_data = tree.export_graphviz(clf, out_file='Dtree_airport_1.dot', feature_names=X_train_ap.columns)
Y_pred_ap = clf.predict(X_test_ap) 
print("Train Accuracy:", metrics.accuracy_score(Y_train_ap, clf.predict(X_train_ap)))
print("Test Accuracy:", metrics.accuracy_score(Y_test_ap, Y_pred_ap))
print("Test Error Rate:",1-metrics.accuracy_score(Y_test_ap, Y_pred_ap))
print("Confusion Matrix:", metrics.confusion_matrix(Y_test_ap, Y_pred_ap)) 
print(classification_report(Y_test_ap, Y_pred_ap))

# Tree Model 2: change tree parameters 
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5, max_features=None, max_leaf_nodes=None, min_samples_leaf=3, min_samples_split=3, min_weight_fraction_leaf=0.0, presort=False, random_state=100, splitter='best')
clf = clf.fit(X_train_ap, Y_train_ap) 

# export estimated tree into dot graphic file
dot_data = tree.export_graphviz(clf, out_file='Dtree_airport_2.dot', feature_names=X_train_ap.columns)
Y_pred_ap = clf.predict(X_test_ap) 
print("Train Accuracy:", metrics.accuracy_score(Y_train_ap, clf.predict(X_train_ap)))
print("Test Accuracy:", metrics.accuracy_score(Y_test_ap, Y_pred_ap))
print("Test Error Rate:",1-metrics.accuracy_score(Y_test_ap, Y_pred_ap))
print("Confusion Matrix:", metrics.confusion_matrix(Y_test_ap, Y_pred_ap)) 
print(classification_report(Y_test_ap, Y_pred_ap))


### SVM model

# SVM model 1
# Normalize data
scaler = StandardScaler()  
scaler.fit(X_train_ap)
X_train_ap = scaler.transform(X_train_ap)  
X_test_ap = scaler.transform(X_test_ap) 
svclassifier = SVC(kernel='linear')    	
svclassifier.fit(X_train_ap, Y_train_ap)  
y_pred_ap = svclassifier.predict(X_test_ap)  	
print(metrics.confusion_matrix(Y_test_ap, y_pred_ap)) 	
print(classification_report(Y_test_ap, y_pred_ap))
print("Train Accuracy:", metrics.accuracy_score(Y_train_ap, svclassifier.predict(X_train_ap)))
print("Test Accuracy:",metrics.accuracy_score(Y_test_ap, y_pred_ap))
print("Test Error Rate:",1-metrics.accuracy_score(Y_test_ap, y_pred_ap))
print("Confusion Matrix:", metrics.confusion_matrix(Y_test_ap, Y_pred_ap))

# SVM model 2
# Change parameters: use 'poly' as kernel
scaler = StandardScaler()  
scaler.fit(X_train_ap)
X_train_ap = scaler.transform(X_train_ap)  
X_test_ap = scaler.transform(X_test_ap) 
svclassifier = SVC(kernel='poly', degree=3)    	
svclassifier.fit(X_train_ap, Y_train_ap)  
y_pred_ap = svclassifier.predict(X_test_ap)  	
print(metrics.confusion_matrix(Y_test_ap, y_pred_ap)) 	
print(classification_report(Y_test_ap, y_pred_ap))
print("Train Accuracy:", metrics.accuracy_score(Y_train_ap, svclassifier.predict(X_train_ap)))
print("Test Accuracy:",metrics.accuracy_score(Y_test_ap, y_pred_ap))
print("Test Error Rate:",1-metrics.accuracy_score(Y_test_ap, y_pred_ap))
print("Confusion Matrix:", metrics.confusion_matrix(Y_test_ap, Y_pred_ap))

 # SVM model 3
# Change parameters: use 'poly' as kernel
scaler = StandardScaler()  
scaler.fit(X_train_ap)
X_train_ap = scaler.transform(X_train_ap)  
X_test_ap = scaler.transform(X_test_ap) 
svclassifier = SVC(kernel='rbf')    	
svclassifier.fit(X_train_ap, Y_train_ap)  
y_pred_ap = svclassifier.predict(X_test_ap)  	
print(metrics.confusion_matrix(Y_test_ap, y_pred_ap)) 	
print(classification_report(Y_test_ap, y_pred_ap))
print("Train Accuracy:", metrics.accuracy_score(Y_train_ap, svclassifier.predict(X_train_ap)))
print("Test Accuracy:",metrics.accuracy_score(Y_test_ap, y_pred_ap))
print("Test Error Rate:",1-metrics.accuracy_score(Y_test_ap, y_pred_ap))
print("Confusion Matrix:", metrics.confusion_matrix(Y_test_ap, Y_pred_ap)) 
  
 

#### Airline modeling
## data preprocessing
train_al = pd.read_csv("train_airline.csv", sep=',', index_col=0)
test_al = pd.read_csv("test_airline.csv", sep=',', index_col=0 )


X_train_al = train_al.drop(['Airline'],axis = 1)
Y_train_al = train_al['Airline']

X_test_al = test_al.drop(['Airline'],axis = 1)
Y_test_al = test_al['Airline']

X_train_al["Nationality"] = X_train_al["Nationality"].astype('category').cat.codes
X_train_al["TripPurpose"] = X_train_al["TripPurpose"].astype('category').cat.codes
X_train_al["FrequentFlightDestination"] = X_train_al["FrequentFlightDestination"].astype('category').cat.codes
X_train_al["SeatClass"] = X_train_al["SeatClass"].astype('category').cat.codes
X_train_al["Income"] = X_train_al["Income"].astype('category').cat.codes
X_train_al["ProvinceResidence"] = X_train_al["ProvinceResidence"].astype('category').cat.codes
X_train_al["ModeTransport"] = X_train_al["ModeTransport"].astype('category').cat.codes


X_test_al["Nationality"] = X_test_al["Nationality"].astype('category').cat.codes
X_test_al["TripPurpose"] = X_test_al["TripPurpose"].astype('category').cat.codes
X_test_al["FrequentFlightDestination"] = X_test_al["FrequentFlightDestination"].astype('category').cat.codes
X_test_al["SeatClass"] = X_test_al["SeatClass"].astype('category').cat.codes
X_test_al["Income"] = X_test_al["Income"].astype('category').cat.codes
X_test_al["ProvinceResidence"] = X_test_al["ProvinceResidence"].astype('category').cat.codes
X_test_al["ModeTransport"] = X_test_al["ModeTransport"].astype('category').cat.codes


### Tree Model 1
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10, max_features=None, max_leaf_nodes=None, min_samples_leaf=10, min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False, random_state=100, splitter='best')
clf = clf.fit(X_train_al, Y_train_al)

# export estimated tree into dot graphic file
dot_data = tree.export_graphviz(clf, out_file='Dtree_airline_1.dot', feature_names=X_train_al.columns)
Y_pred_al = clf.predict(X_test_al) 
print("Train Accuracy:", metrics.accuracy_score(Y_train_al, clf.predict(X_train_al)))
print("Test Accuracy:", metrics.accuracy_score(Y_test_al, Y_pred_al))
print("Test Error Rate:",1-metrics.accuracy_score(Y_test_al, Y_pred_al))
print("Confusion Matrix:", metrics.confusion_matrix(Y_test_al, Y_pred_al)) 
print(classification_report(Y_test_al, Y_pred_al))

# Tree Model 2: change tree parameters 
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5, max_features=None, max_leaf_nodes=None, min_samples_leaf=3, min_samples_split=3, min_weight_fraction_leaf=0.0, presort=False, random_state=100, splitter='best')
clf = clf.fit(X_train_al, Y_train_al)

# export estimated tree into dot graphic file
dot_data = tree.export_graphviz(clf, out_file='Dtree_airline_2.dot', feature_names=X_train_al.columns)
Y_pred_al = clf.predict(X_test_al) 
print("Train Accuracy:", metrics.accuracy_score(Y_train_al, clf.predict(X_train_al)))
print("Test Accuracy:", metrics.accuracy_score(Y_test_al, Y_pred_al))
print("Test Error Rate:",1-metrics.accuracy_score(Y_test_al, Y_pred_al))
print("Confusion Matrix:", metrics.confusion_matrix(Y_test_al, Y_pred_al)) 
print(classification_report(Y_test_al, Y_pred_al))

### SVM model
# SVM model 1:use linear kernel
scaler = StandardScaler()  
scaler.fit(X_train_al)
X_train_al = scaler.transform(X_train_al)  
X_test_al = scaler.transform(X_test_al) 
svclassifier = SVC(kernel='linear')    	
svclassifier.fit(X_train_al, Y_train_al)  
y_pred_al = svclassifier.predict(X_test_al)  	
print(metrics.confusion_matrix(Y_test_al, y_pred_al)) 	
print(classification_report(Y_test_al, y_pred_al))
print("Train Accuracy:", metrics.accuracy_score(Y_train_al, svclassifier.predict(X_train_al)))
print("Test Accuracy:",metrics.accuracy_score(Y_test_al, y_pred_al))
print("Test Error Rate:",1-metrics.accuracy_score(Y_test_al, y_pred_al))
print("Confusion Matrix:", metrics.confusion_matrix(Y_test_al, Y_pred_al))

# SVM model 2:use poly kernel
scaler = StandardScaler()  
scaler.fit(X_train_al)
X_train_al = scaler.transform(X_train_al)  
X_test_al = scaler.transform(X_test_al) 
svclassifier = SVC(kernel='poly', degree=3)    	
svclassifier.fit(X_train_al, Y_train_al)  
y_pred_al = svclassifier.predict(X_test_al)  	
print(metrics.confusion_matrix(Y_test_al, y_pred_al)) 	
print(classification_report(Y_test_al, y_pred_al))
print("Train Accuracy:", metrics.accuracy_score(Y_train_al, svclassifier.predict(X_train_al)))
print("Test Accuracy:",metrics.accuracy_score(Y_test_al, y_pred_al))
print("Test Error Rate:",1-metrics.accuracy_score(Y_test_al, y_pred_al))
print("Confusion Matrix:", metrics.confusion_matrix(Y_test_al, Y_pred_al))

# SVM model 3:use rbf kernel
scaler = StandardScaler()  
scaler.fit(X_train_al)
X_train_al = scaler.transform(X_train_al)  
X_test_al = scaler.transform(X_test_al) 
svclassifier = SVC(kernel='rbf')    	
svclassifier.fit(X_train_al, Y_train_al)  
y_pred_al = svclassifier.predict(X_test_al)  	
print(metrics.confusion_matrix(Y_test_al, y_pred_al)) 	
print(classification_report(Y_test_al, y_pred_al))
print("Train Accuracy:", metrics.accuracy_score(Y_train_al, svclassifier.predict(X_train_al)))
print("Test Accuracy:",metrics.accuracy_score(Y_test_al, y_pred_al))
print("Test Error Rate:",1-metrics.accuracy_score(Y_test_al, y_pred_al))
print("Confusion Matrix:", metrics.confusion_matrix(Y_test_al, Y_pred_al))
 

