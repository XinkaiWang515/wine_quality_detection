from rf_param_tune import rf_para_tune
# from data import r_X_train, r_y_train, r_X_val, r_y_val, r_X_test, r_y_test, scal_X_train, scal_X_val, scal_X_test
from data_adjust import r_X_train, r_y_train, r_X_val, r_y_val, r_X_test, r_y_test, scal_X_train, scal_X_val, scal_X_test
from feature_sel import feature_selection
from sklearn.ensemble import RandomForestClassifier
from smote import X_res,y_res
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, make_scorer

def random_forest(X_train,y_train,X_val,X_test,y_test):
    rf_best_param, bal_acc, val_accu = rf_para_tune(X_train,y_train,X_val)
    features = feature_selection(X_train,y_train,rf_best_param['fea_num'])[-1]
    sel_X_train=X_train.iloc[:,features]
    sel_X_test=X_test.iloc[:,features]
    rf = RandomForestClassifier(n_estimators=rf_best_param['n_estimators'],max_features=rf_best_param['max_features'],
                                max_depth=rf_best_param['max_depth'],min_samples_leaf=rf_best_param['min_samples_leaf'],
                                min_samples_split=rf_best_param['min_samples_split'],class_weight='balanced',random_state=1)
    rf.fit(sel_X_train,y_train)
    train_pred = rf.predict(sel_X_train)
    tr_accu = accuracy_score(y_train, train_pred)
    tr_bal_accu = balanced_accuracy_score(y_train, train_pred)

    y_pred = rf.predict(sel_X_test)
    feature_n = rf_best_param['fea_num']
    bal_accu = balanced_accuracy_score(y_test,y_pred)
    f1_sc = f1_score(y_test,y_pred,average='weighted')
    accu = accuracy_score(y_test,y_pred)

    return feature_n,bal_accu,f1_sc,accu,tr_bal_accu,tr_accu

if __name__=="__main__":
    feature_n, bal_accu, f1_sc, accu,bal_acc,tr_accu = random_forest(r_X_train,r_y_train,r_X_val,r_X_test,r_y_test)
    print('selected feature numbers: %d'% feature_n)
    print('random forest test balanced accuracy score:', bal_accu)
    print('random forest test f1 score:', f1_sc)
    print('random forest test accuracy:', accu)
    print('random forest training balanced accuracy:', bal_acc)
    print('random forest training accuracy:', tr_accu)


    std_feature_n, std_bal_accu, std_f1_sc, std_accu,std_bal_acc,std_tr_accu = random_forest(scal_X_train,r_y_train,scal_X_val,scal_X_test,r_y_test)
    print('\n')
    print('after standardizing features')
    print('number of selected features: %d'% std_feature_n)
    print('random forest test balanced accuracy score:', std_bal_accu)
    print('random forest test f1 score:', std_f1_sc)
    print('random forest test accuracy:', std_accu)
    print('random forest training balanced accuracy:', std_bal_acc)
    print('random forest training accuracy:', std_tr_accu)

    sm_feature_n,sm_bal_accu,sm_f1_sc,sm_accu,sm_bal_acc,sm_tr_accu = random_forest(X_res, y_res, r_X_val, r_X_test, r_y_test)
    print('\n')
    print('after downsampling by OSS and then use SMOTE')
    print('number of selected features: %d' % sm_feature_n)
    print('random forest test balanced accuracy score:', sm_bal_accu)
    print('random forest test f1 score:', sm_f1_sc)
    print('random forest test accuracy:', sm_accu)
    print('random forest training balanced accuracy:', sm_bal_acc)
    print('random forest training accuracy:', sm_tr_accu)

