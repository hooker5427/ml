from sklearn.linear_model import LinearRegression
from  sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

x=1.0/(np.arange(1,11)+np.arange(0,10)[:,np.newaxis]) #10*10数组
y=np.ones(10)
print(x) #x矩阵
print(y) #数组

n_alphas=200
alphas=np.logspace(-10,-2,n_alphas) #指数切割
print(alphas)
clf=linear_model.Ridge(fit_intercept=False) #创建令回归的对象

coefs=[]#斜率的集合 ，每个一个alpha,尝试一下，
for  alpha  in alphas:
    clf.set_params(alpha=alpha)
    clf.fit(x,y)
    coefs.append(clf.coef_)#插入斜率

#展示 ，绘图
plt.figure(figsize=(12,9 ))
ax=plt.gca()

ax.plot(alphas,coefs)
ax.set_xscale("log")
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel("alpha")
plt.ylabel("weights")
plt.title("回归曲线")
plt.axis("tight")
plt.show()

