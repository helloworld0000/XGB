import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation,metrics
from sklearn.grid_search import GridSearchCV
import socre
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12,4

#load the clean data here

train = pd.read_csv('cleandata/train.csv')
test = pd.read_csv('cleandata/test.csv')
pre = pd.read_csv('cleandata/pre.csv')

testmondey = pre.iloc[:,0:2]
testmondey.columns = ['shopid','target']
# print testmondey
train = (pd.merge(train.reset_index(),testmondey,on='shopid')).set_index('shopid')

print train.shape
print test.shape
target = 'target'
IDcol = 'shopid'
# print type(train[1:1])
train[target] = train[target]/3000
# print train[target].max(axis=0)

#the test dataframe
testtarget = pre.iloc[:,[0,8]]
testtarget.columns = ['shopid','target']
test = (pd.merge(test.reset_index(),testtarget,on='shopid')).set_index('shopid')
test[target] = test[target]/3000

def modelfit(alg,dtrain,dtest,predictors,useTrainCV=True,cv_folds=5,early_stopping_rounds=50):

    if(useTrainCV):
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values,label= (dtrain[target].values))
        cvresult = xgb.cv(xgb_param,xgtrain,num_boost_round=alg.get_params()['n_estimators'],nfold=cv_folds,metrics='auc',early_stopping_rounds=early_stopping_rounds)
        print cvresult.shape[0]
        alg.set_params(n_estimators = cvresult.shape[0])
    #fit on the data
    # print dtest[testvalues]
    print 'fitting now'
    print dtrain[predictors].shape
    print dtrain[target].shape
    alg.fit(dtrain[predictors],dtrain[target],eval_metric='auc')
    print 'fitting ok'
    #predict training set:
    return alg

list = []
list.append(target)
list.append(IDcol)
# for i in range(0,122):
#     list.append('city_'+str(i))
for i in range(0,44):
    list.append('cate3_'+str(i))
print list
# for i in range(1,8):
#     list.append('1-'+str(i))
# for i in range(1,8):
#     list.append('2-'+str(i))
# print list

predictors = [x for x in train.columns if x not in list]
print len(predictors)
# testvalues = ['shopid','2016-10-11','2016-10-12','2016-10-13','2016-10-14','2016-10-15','2016-10-16','2016-10-17']
# wantdata = ['shopid','2016-09-20','2016-09-21','2016-09-22','2016-09-23','2016-09-24','2016-09-25','2016-09-26']
# testvalues = ['2016-10-11']

# xgb1 = XGBClassifier(
#
#     learning_rate=0.1,
#     n_estimators=1000,
#     max_depth=5,
#     min_child_weight=3,
#     gamma=0,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     objective='binary:logistic',
#     scale_pos_weight=1,
#     seed=27
# )
# alg = modelfit(xgb1,train,test,predictors)

param_test = {
    'max_depth':range(3,10,2),
    'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator= XGBClassifier(learning_rate=0.1,n_estimators=184,max_depth=5,min_child_weight=1,gamma=0,
                                                 subsample=0.8,colsample_bytree=0.8,objective='binary:logistic',scale_pos_weight=1,
                                                 seed=27),param_grid = param_test,scoring='roc_auc',n_jobs=4,iid=False,cv = 5)
alg = gsearch1.fit(train[predictors],train[target])
print gsearch1.grid_scores_,gsearch1.best_params_,gsearch1.best_score_



#
# dtrain_predictions = alg.predict(train[predictors])
#
# dtrain_predictions = dtrain_predictions * 3000
# print dtrain_predictions
# train[target] = train[target] * 3000
# print train[target].values
#
# # dtrain_predprob = alg.predict_proba(train[predictors])[:, 1]
# # print the modelreport:
#
# print '\ntrain Model Report'
#
# # print "Accuracy: %.4g" %metrics.accuracy_score(dtrain[target].values, dtrain_predictions)
# # print 'AUC SCORE(Train):%f' %metrics.roc_auc_score(dtrain[target].values,dtrain_predprob)
#
# thescroe = socre.calculate_score(train[target].values, dtrain_predictions,'traing')
# print thescroe
#
# feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
# feat_imp.plot(kind='bar', title='Feature Important')
# plt.ylabel('Feature Importance Score train')
# plt.show()
#
# dtest_predictions = alg.predict(test[predictors])
# print '\n test Model Report'
# dtest__predictions = dtest_predictions * 3000
# test[target] = test[target] * 3000
# print dtest__predictions
# print test[target].values
# thescroe = socre.calculate_score(test[target].values, dtest_predictions,'testg')
# print thescroe






