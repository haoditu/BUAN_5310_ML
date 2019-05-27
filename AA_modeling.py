import os
import pandas as pd
import statsmodels.api as sm


os.chdir("/Users/chelseaj/Documents/GitHub/BUAN_5310_ML")

acs = pd.read_csv("acs.csv", sep=',', header=0)

# Summary statistics
pd.set_option('display.max_columns', None)
acs.describe()

# Correlation
acs.corr()

Y = acs['Airport'].astype('category').cat.codes
X = acs.drop(['Airport'],axis = 1)

# Logit model 
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

