import pandas as pd
import socre
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12,4

pre = pd.read_csv('result/test_pre.csv')
real = pd.read_csv('result/test_real.csv')
real['target'] = real['target']*3000

if (len(pre) != len(real)):
    print 'len(pre)!=len(real)', '\n'
if (len(pre.columns) != len(real.columns)):
    print 'len(pre.columns)!=len(real.columns)', '\n'
N = len(pre)
T = len(pre.columns)
print 'N:', N, '\t', 'T:', T, '\n'

n = 0
t = 0
L = 0

while (t < T):
    n = 0
    while (n < N):
        c_it = round(pre.ix[n, t])
        c_git = round(real.ix[n, t])

        if (c_it == 0 and c_git == 0):
            c_it = 1
            c_git = 1

        L = L + abs((float(c_it) - c_git) / (c_it + c_git))
        n = n + 1
    t = t + 1
# print L
print L / (N * T)

pre.columns= ['shopid','target']
real.columns= ['shopid','real']

all = (pd.merge(pre,real,on='shopid')).set_index('shopid')
print all

# all.columns = ['target','real']
# all.plot(x='shopid',y='target',color='DarkBlue')
# all.plot(x='shopid',y='real',color='Red')
# plt.show()

all.cumsum()
plt.figure()
all.plot()
plt.show()