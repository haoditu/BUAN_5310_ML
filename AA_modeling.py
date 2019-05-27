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
