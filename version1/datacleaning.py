import pandas as pd
import numpy as np


#first load the data from the csvfile
def getothervalues(thedata,thedata_weekend):
    # get some good values add on it add on the train
    thedata_sum = thedata.sum(axis=1)
    thedata_mean = thedata.mean(axis=1)
    thedata_median = thedata.median(axis=1)
    thedata_std = thedata.std(axis=1)  # std
    thedata_ratio_wk = (thedata[thedata_weekend]).sum(axis=1) / (thedata_sum.replace(0, 1))
    thedata['sumABCD'] = thedata_sum
    # train_x['meanABCD'] = train_mean
    # train_x['maxABCD'] = train_max
    thedata['medianABCD'] = thedata_median
    thedata['stdABCD'] = thedata_std
    thedata['ratio_wk'] = thedata_ratio_wk
    # train_y = c.set_index('shopid')  # use C to do it
    return thedata


a = pd.read_csv('data/A.csv')
b = pd.read_csv('data/B.csv')
c = pd.read_csv('data/C.csv')
d = pd.read_csv('data/D.csv')
shopinfo = pd.read_csv('data/shopinfo.csv')


#get the train and test data
#here I want a+b to be as the train data
#b+c to be the test data
#c+d to be the predict data

train = (pd.merge(a,b,on='shopid')).set_index('shopid')
test = (pd.merge(b,c,on='shopid')).set_index('shopid')
predict = (pd.merge(c,d,on='shopid')).set_index('shopid')

train_weekend = ['2016-09-24', '2016-09-25', '2016-10-15', '2016-10-16']
test_weekend = ['2016-10-15','2016-10-16','2016-10-22','2016-10-23']
predict_weekend = ['2016-10-22','2016-10-23','2016-10-29','2016-10-30']



#get the all data
trainall = getothervalues(train,train_weekend)
testall = getothervalues(test,test_weekend)
predictall = getothervalues(predict,predict_weekend)

#merge with the shopinfo
train_mat = (pd.merge(trainall.reset_index(),shopinfo,on='shopid')).set_index('shopid')
test_mat = (pd.merge(testall.reset_index(),shopinfo,on='shopid')).set_index('shopid')
pre_mat = (pd.merge(predictall.reset_index(),shopinfo,on='shopid')).set_index('shopid')

#save the clean data to csvfile
train_mat.to_csv('cleandata/train.csv')
test_mat.to_csv('cleandata/test.csv')
pre_mat.to_csv('cleandata/pre.csv')









