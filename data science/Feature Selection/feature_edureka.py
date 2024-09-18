
"""Importing the libraries"""
import pandas as pd
import numpy as np

"""Loading the data"""
data = "diabetes.csv"
features = ['preg','plas','pres','skin','test','mass','pedl','age','class']
df = pd.read_csv(data, names=features)

# print(df.head())
# print(df.shape)

"""Predicting the data"""
data = df.values
X = data[:,0:8]
Y = data[:,0]


"""Filter Method"""
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Feature extraction
chi_best = SelectKBest(score_func=chi2, k=4)
k_best = chi_best.fit(X,Y)

# Summary Score
np.set_printoptions(precision=3)
# print(k_best.scores_)

# Summarize selected features
k_features = k_best.transform(X)
# print(k_features[0:5,:])









"""Wrapper Method"""
# Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

# Feature extraction
model_lr = LogisticRegression()
recur_fe = RFE(model_lr, n_features_to_select=3)
feature = recur_fe.fit(X, Y)
# print(f"Number of features: {feature.n_features_}")
# print(f"Support of features: {feature.support_}")
# print(f"Ranking of features: {feature.ranking_}")








"""Embedded Method"""
# Ridge Regression / L2 Regularization
from sklearn.linear_model import Ridge

riddge_reg = Ridge(alpha=1.0)
riddge_reg.fit(X, Y)

# A helper function to printing the coefficients
def print_coeffs(coef, names=None, sort=False):
   if names == None:
      names = ["X%s" % x for x in range(len(coef))]
   lst = zip(coef, names)
   if sort:
      lst = sorted(lst, key= lambda x: -np.abs(x[0]))
   return " + ".join("%s * %s" % (round(coef, 3), name) for coef, name in lst)

print("Ridge mode: ", print_coeffs(riddge_reg.coef_))
