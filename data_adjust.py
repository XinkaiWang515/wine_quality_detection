import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

red = pd.read_csv('winequality-red.csv')
white = pd.read_csv('winequality-white.csv')

white = white[white.quality.isin([4,5,6,7,8])]
r_X = white.iloc[:,:-1]
r_y = white.iloc[:,-1]
# red = red[red.quality.isin([4,5,6,7])]
# r_X = red.iloc[:,:-1]
# r_y = red.iloc[:,-1]

r_X_training, r_X_test, r_y_training, r_y_test = train_test_split(r_X, r_y, test_size=0.2, random_state=42)
r_X_train, r_X_val, r_y_train, r_y_val = train_test_split(r_X_training, r_y_training, test_size=0.2, random_state=42)

scaler = StandardScaler()
scal_X_train = scaler.fit_transform(r_X_train)
columns = list(red)[:-1]
scal_X_train = pd.DataFrame(scal_X_train,columns=columns)
scal_X_val = scaler.fit_transform(r_X_val)
scal_X_val = pd.DataFrame(scal_X_val,columns=columns)
scal_X_test = scaler.fit_transform(r_X_test)
scal_X_test = pd.DataFrame(scal_X_test,columns=columns)

if __name__=="__main__":
    print(red['quality'].value_counts())