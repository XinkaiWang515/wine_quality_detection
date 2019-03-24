import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

red = pd.read_csv('winequality-red.csv')
white = pd.read_csv('winequality-white.csv')
# r_X = white.iloc[:,:-1]
# r_y = white.iloc[:,-1]
r_X = red.iloc[:,:-1]
r_y = red.iloc[:,-1]

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
    # print(red['quality'].value_counts())
    for l in list(red)[:-1]:
        print(white[l].describe())
    # print(r_y_train.head(5))
    # print(type(r_y_train))