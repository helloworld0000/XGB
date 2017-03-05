

import pandas
def calculate_score(pre, real,save):
    if (len(pre.shape) == 1):
        pre = pandas.DataFrame(pre, columns=[0])
        real = pandas.DataFrame(real, columns=[0])
    else:
        pre = pandas.DataFrame(pre, columns=[i for i in range(pre.shape[1])])
        real = pandas.DataFrame(real, columns=[i for i in range(real.shape[1])])

    pre.to_csv('result/'+save+'_pre.csv')
    real.to_csv('result/' + save + '_real.csv')

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
    return L / (N * T)