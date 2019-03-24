from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import OneSidedSelection
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from data import red,white
# from data_adjust import red,white
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# red = pd.read_csv('winequality-red.csv')
# white = pd.read_csv('winequality-white.csv')

# Since the largest class ('quality'=5) has 681 data while the smallest class ('quality'=3) has only 10 data, after
#  splitting data into train, val and test, I randomly sampled class 5 and 6 in scale of 250 samples, then used SMOTE
# to oversample other classes.

# training, test = train_test_split(white,test_size=0.2,random_state=42)
training, test = train_test_split(red,test_size=0.2,random_state=42)

train, val = train_test_split(training,test_size=0.2,random_state=42)
X_train = train.iloc[:,:-1]
y_train = train.iloc[:,-1]
# print(X_train.head(5))

oss = OneSidedSelection(random_state=42)
X_und, y_und = oss.fit_sample(X_train, y_train)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(X_und, y_und)
columns = list(red)[:-1]
X_res = pd.DataFrame(X_res,columns=columns)
y_res = pd.Series(y_res)
# print(X_res.shape)


# training, test = train_test_split(red,test_size=0.2,random_state=42)
# train, val = train_test_split(training,test_size=0.2,random_state=42)
# qua5 = train.loc[train['quality'] == 5]
# qua6 = train.loc[train['quality'] == 6]
# qua_other = train.loc[train['quality'].isin([3,4,7,8])]
# q5 = qua5.sample(n=250)
# q6 = qua6.sample(n=250)
# new_tr = pd.concat([q5,q6,qua_other])
# new_tr = shuffle(new_tr)
# X_train = new_tr.iloc[:,:-1]
# y_train = new_tr.iloc[:,-1]
# # print(X_train.head(5))
#
# sm = SMOTE(random_state=42)
# X_res, y_res = sm.fit_sample(X_train, y_train)
# columns = list(red)[:-1]
# X_res = pd.DataFrame(X_res,columns=columns)
# y_res = pd.Series(y_res)
# # print(X_res.shape)

# print(new_tr.head(10))
if __name__=="__main__":
    # print(train['quality'].value_counts())
    print(X_res.shape)
    print(X_res.head(5))
    print(y_res.shape)
