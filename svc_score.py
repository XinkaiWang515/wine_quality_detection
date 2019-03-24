# from data import r_X_training, r_y_training, r_X_train, r_y_train, r_X_val, r_y_val
from data_adjust import r_X_test, r_y_test, r_X_train, r_y_train, r_X_val, r_y_val
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, make_scorer
import numpy as np
from feature_sel import feature_selection
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

svc = SVC(kernel='rbf',gamma='auto',C=10)
svc.fit(r_X_train,r_y_train)
pred_y = svc.predict(r_X_test)
bal_accu = balanced_accuracy_score(r_y_test,pred_y)
accu = accuracy_score(r_y_test,pred_y)
f1_sc = f1_score(r_y_test,pred_y,average='weighted')

if __name__=="__main__":
    # print('selected feature numbers: %d' % feature_n)
    print('random forest test balanced accuracy score:', bal_accu)
    print('random forest test f1 score:', f1_sc)
    print('random forest test accuracy:', accu)
    # print('random forest training balanced accuracy:', bal_acc)
    # print('random forest training accuracy:', tr_accu)