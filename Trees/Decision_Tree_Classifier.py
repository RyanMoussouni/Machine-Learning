#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:26:31 2020

@author: moussouni
"""

#%% Packages
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
from tp_trees_aux import rand_checkers,frontiere,plot_2d
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV,learning_curve
from sklearn.utils import shuffle
from scipy.stats import binom
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

#%% Q2
tr_err_gini = []
tr_err_entropy = []
data, y = rand_checkers(n1 = 228, n2 = 228) #important that we don't get new data at each iteration
for k in range(2,16):
    gini_tree = DecisionTreeClassifier(criterion = "gini",max_depth = k)
    entropy_tree = DecisionTreeClassifier(criterion = "entropy",max_depth = k)
    gini_tree.fit(data,y)
    entropy_tree.fit(data,y)
    tr_err_gini.append(1-gini_tree.score(data,y))
    tr_err_entropy.append(1-entropy_tree.score(data,y))
plt.figure()
plt.title('Train error for both gini and entr.')
plt.plot(range(2,16), tr_err_gini,'k--', label='gini',color = 'blue')
plt.plot(range(2,16), tr_err_entropy,'k--', label='entropy',color = 'red')
plt.legend()

'''
Training error seems to get better as the depth of the trees increases.
Surely a high score on profound trees can be misleading as it can lead to overfit. 
In the following, we'll take 10 as the optimum depth.
'''


#%% Decision function for the Decision Tree Classifier
entropy_tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 10)
entropy_tree.fit(data, y)
plt.figure()
plot_2d(data,y)
frontiere(entropy_tree.predict, data)

#%% Computing new distrib and checking results
data, y = rand_checkers(n1 = 80, n2 = 80)
plt.figure()
plot_2d(data,y)
frontiere(entropy_tree.predict, data)
print("Error on this new data : %f" %1-entropy_tree.score(data,y))

'''
We get a 15% error on this new data, compared to the training error, that's pretty high, our tree might be overfit.
'''

#%% With digits
digits = load_digits()
data, y = shuffle(digits['data'],digits['target'])
param_grid = { 'max_depth' : range(2,16)}
tree = DecisionTreeClassifier()
grid = GridSearchCV(tree, param_grid)
grid.fit(data,y)
print(grid.best_estimator_)

#%% Learning curve
#Getting the curve
opt_tree = DecisionTreeClassifier(max_depth = 10)
train_sizes, tr_sc, tst_sc, fit_times,score_times = learning_curve(tree, data, y,return_times = True)
#Plotting it
fig, axes = plt.subplots(2,2)
axes[0][0].plot(train_sizes, tr_sc.mean(axis = 1))
axes[0][1].plot(train_sizes, tst_sc.mean(axis = 1))
axes[1][0].plot(train_sizes, fit_times.mean(axis = 1))
axes[1][1].plot(train_sizes, score_times.mean(axis = 1))

#%% Binom
# Constantes
L = 10
p = 0.7
# Binomiale
rv = binom(L,p)
success = rv.pmf(5) + rv.pmf(6) + rv.pmf(7) + rv.pmf(8) + rv.pmf(9) + rv.pmf(10)
print('probability of success %f' %success)

#%% Random forests
# Parameters
n_estimators = 10
plot_colors = "bry"
plot_step = 0.02
# Load data
iris = load_iris()
X_unscaled, y = iris.data[:, :2], iris.target
# Standardize
X = preprocessing.scale(X_unscaled)
# RF fitting
model = RandomForestClassifier(n_estimators=n_estimators)
clf = model.fit(X, y)
# Plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
plt.figure()
ypred = []
for tree in model.estimators_:
	ypred.append(tree.predict(np.c_[xx.ravel(), yy.ravel()]))
ypred = np.array(ypred)
Z = ypred.mean(axis = 0)
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, alpha=1. / n_estimators, cmap=plt.cm.Paired)
plt.axis("tight")
# Plot the training points
for i, c in zip(range(3), plot_colors):
	idx = np.where(y == i)
	plt.scatter(X[idx, 0], X[idx, 1], c=c,label=iris.target_names[i],cmap=plt.cm.Paired)
plt.legend(scatterpoints=1)
plt.show()

#%% Comparing Decision Tree and Random Forest
#Load data
iris = load_iris()
X_unscaled, y = iris.data[:, :2], iris.target
# Standardize
X = preprocessing.scale(X_unscaled)

#param_grid
param_grid = {'max_depth' : range(1,30)}

#score
rf_score = []
dt_score = []
for depth in param_grid['max_depth']:
    rf = RandomForestClassifier(n_estimators = 14, max_depth = depth)
    dt = DecisionTreeClassifier(max_depth = depth)
    rf_score.append(cross_val_score(rf,X,y,cv=6).mean())
    dt_score.append(cross_val_score(dt,X,y,cv=6).mean())
plt.plot(range(1,30),rf_score,color = 'red')
plt.plot(range(1,30),dt_score, color = 'blue')
''' 
Des resultats similaires
'''
#%% Feature importance
