from data import r_X_train, r_y_train, r_X_val, r_y_val, r_X_test, r_y_test, scal_X_train, scal_X_val, scal_X_test
# from data_adjust import r_X_train, r_y_train, r_X_val, r_y_val, r_X_test, r_y_test, scal_X_train, scal_X_val, scal_X_test
from feature_sel import feature_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import numpy as np
from sklearn.metrics import roc_auc_score
from lrg_param_tune import lrg_para_tune
from smote import X_res,y_res
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def lrg_accuracy(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    tr_pred = model.predict(X_train)
    tr_accu = accuracy_score(y_train,tr_pred)
    tr_bal_accu = balanced_accuracy_score(y_train,tr_pred)

    pred = model.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, pred)
    acc = accuracy_score(y_test, pred)
    f1_sc = f1_score(y_test, pred, average='weighted')

    return bal_acc,acc,f1_sc,tr_bal_accu,tr_accu

def logistic_reg(X_train, y_train, X_val, X_test, y_test):
    best_param,accu,bal_accu = lrg_para_tune(X_train, y_train, X_val)
    features = feature_selection(X_train, y_train, best_param['fea_num'])[1]
    sel_X_train = X_train.iloc[:, features]
    sel_X_test = X_test.iloc[:, features]

    mul_lrg = LogisticRegression(multi_class='multinomial', solver=best_param['solver'], C=best_param['C'],
                                 class_weight='balanced')
    bal_acc_lrg, acc_lrg, f1_lrg, tr_bal_accu,tr_accu = lrg_accuracy(mul_lrg, sel_X_train, y_train, sel_X_test, y_test)

    # mul_lrg_fea_sel = LogisticRegression(multi_class='multinomial', solver='newton-cg')
    # acc_mul_lrg_fea_sel = lrg_accuracy(mul_lrg_fea_sel,sel_X_train,y_train,sel_X_val,y_val)

    return bal_acc_lrg, acc_lrg, f1_lrg, tr_bal_accu,tr_accu, best_param['fea_num']


if __name__=="__main__":
    bal_acc_lrg, acc_lrg, f1_lrg, tr_bal_accu,tr_accu,feature_n = logistic_reg(r_X_train, r_y_train, r_X_val, r_X_test, r_y_test)
    print('number of selected features is: %d'% feature_n)
    print('multinomial Logistic Regression test balanced accuracy:', bal_acc_lrg)
    print('multinomial Logistic Regression test f1 score:', f1_lrg)
    print('multinomial Logistic Regression test accuracy:', acc_lrg)
    print('best training model balanced accuracy:', tr_bal_accu)
    print('best training model accuracy:', tr_accu)

    std_bal_acc_lrg,std_acc_lrg,std_f1_lrg,std_tr_bal_accu,std_tr_accu,std_feature_n = logistic_reg(scal_X_train, r_y_train, scal_X_val, scal_X_test, r_y_test)
    print('\n')
    print('after standardizing features')
    print('number of selected features is: %d' % std_feature_n)
    print('multinomial Logistic Regression test balanced accuracy:', std_bal_acc_lrg)
    print('multinomial Logistic Regression test f1 score:', std_f1_lrg)
    print('multinomial Logistic Regression test accuracy:', std_acc_lrg)
    print('best training model balanced accuracy:', std_tr_bal_accu)
    print('best training model accuracy:', std_tr_accu)

    sm_bal_acc_lrg, sm_acc_lrg, sm_f1_lrg, sm_tr_bal_accu, sm_tr_accu, sm_feature_n = logistic_reg(X_res, y_res, r_X_val, r_X_test, r_y_test)
    print('\n')
    print('after downsampling by OSS and then use SMOTE')
    print('number of selected features is: %d' % sm_feature_n)
    print('multinomial Logistic Regression balanced accuracy:', sm_bal_acc_lrg)
    print('multinomial Logistic Regression f1 score:', sm_f1_lrg)
    print('multinomial Logistic Regression accuracy:', sm_acc_lrg)
    print('best training model balanced accuracy:', sm_tr_bal_accu)
    print('best training model accuracy:', sm_tr_accu)



    # print('multi_LG accuracy after feature selection:', acc_mul_lrg_fea_sel)
