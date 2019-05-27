import pandas as pd
import statsmodels.api as sm

acs = pd.read_csv("acs.csv")

# Summary statistics
pd.set_option('display.max_columns', None)
acs.describe()

# Correlation
acs.corr()


Y = acs['Airport']
X = acs.drop(['Airport'],axis = 1)

# Logit model 
logit_model = sm.Logit(Y, X).fit()
logit_model.summary()
