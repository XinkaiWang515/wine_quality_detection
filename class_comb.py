import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

red = pd.read_csv('winequality-red.csv')
white = pd.read_csv('winequality-white.csv')
r_X = red.iloc[:,:-2]
r_y = red.iloc[:,-1]
y = label_binarize(r_y, classes=[3,4,5,6,7,8])
n_classes = 6
r_X_training, r_X_test, r_y_training, r_y_test = train_test_split(r_X, y, test_size=0.2)

clf = OneVsRestClassifier(LogisticRegression(random_state=0))
y_score = clf.fit(r_X_training, r_y_training).decision_function(r_X_test)

sm = SMOTE(random_state=0)
X_res, y_res = sm.fit_sample(r_X,r_y)
y = label_binarize(r_y, classes=[3,4,5,6,7,8])
r_X_training, r_X_test, r_y_training, r_y_test1 = train_test_split(r_X, y, test_size=0.2)
clf1 = OneVsRestClassifier(LogisticRegression(random_state=0))
y_score1 = clf.fit(X_res, y_res).decision_function(r_X_test)

# fpr = dict()
# tpr = dict()
# roc_auc = dict()
class_es = [3,4,5,6,7,8]
for i in range(n_classes):
    # fpr[i], tpr[i], _ = roc_curve(r_y_test[:, i], y_score[:, i])
    # roc_auc[i] = auc(fpr[i], tpr[i])
    print('no SMOTE AUROC score class %d vs rest:'% class_es[i],roc_auc_score(r_y_test[:, i], y_score[:, i]))
    print('SMOTE AUROC score class %d vs rest:'% class_es[i],roc_auc_score(r_y_test1[:, i], y_score1[:, i]))
    print('\n')

# for i in range(n_classes):
#     plt.figure()
#     plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic example')
#     plt.legend(loc="lower right")
#     plt.show()