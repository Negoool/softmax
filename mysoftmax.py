''' Exercise 12(chapter 4): Softmax Regression'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin

import os
os.system('cls')

from sklearn import datasets
iris = datasets.load_iris()
print iris.keys()
X_iris = iris['data'][:,(2, 3)]# np(150L,4L)
y_iris = iris['target']

np.random.seed(2042)
test_ratio, valid_ratio = .2, .2
test_size = int(len(X_iris)*test_ratio)
valid_size = int(len(X_iris)*valid_ratio)
train_size = len(X_iris) - test_size - valid_size
random_indice = np.random.permutation(len(X_iris))
X_train = X_iris[random_indice[:train_size]]
y_train = y_iris[random_indice[:train_size]]
X_valid = X_iris[random_indice[train_size:-test_size]]
y_valid = y_iris[random_indice[train_size:-test_size]]
X_test = X_iris[random_indice[-test_size:]]
y_test = y_iris[random_indice[-test_size:]]

# from sklearn.preprocessing import LabelBinarizer
# encoder = LabelBinarizer()
# y_iris = encoder.fit_transform(y_init)
def my_LabelBinarizer(y):
    n_class = len(np.unique(y))
    y_one_hot = np.zeros((len(y), n_class))
    y_one_hot[np.arange(len(y)),y ] = 1
    return y_one_hot

y_train_one_hot = my_LabelBinarizer(y_train)
y_test_one_hot = my_LabelBinarizer(y_test)
y_valid_one_hot = my_LabelBinarizer(y_valid)

class MY_Softmax(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha = .1, random_state = 2042, initial_theta = None,
    fit_intercept = True, history = None, max_iter = 1000, landa = .1):
        self.alpha = alpha
        self.random_state = random_state
        self.initial_theta = initial_theta
        self.fit_intercept = fit_intercept
        self.history = history
        self.max_iter = max_iter
        self.landa = landa

    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError('shape of X and y should be tha same')
        np.random.seed(self.random_state)
        X_copy = np.array(X)
        N =  X.shape[0]
        n_class = y.shape[1]

        if self.fit_intercept == True:
            X = np.c_[np.ones((N,1)) , X]
        d = X.shape[1]

        if self.initial_theta is None:
            self.initial_theta = np.random.randn(n_class,d)
        self.theta_ = np.array(self.initial_theta)

        for iteratin in range(self.max_iter):
            p = self.predict_proba(X_copy)
            if self.history is not None:
                self.history.append((self.score(X_copy, y)))
            gradients = ( (1./N)* (p - y).T.dot(X)) +\
             np.c_[np.zeros((3,1)), self.landa* self.theta_[:,1:]]
            self.theta_ = self.theta_ - self.alpha*gradients

        return self

    def predict_proba(self, X, y = None):
        N = X.shape[0]
        if self.fit_intercept == True:
            X = np.c_[np.ones((N,1)) , X]
        s = X.dot(self.theta_.T) #  N*n_class
        row_sums = np.sum(np.exp(s), axis = 1)
        p = np.exp(s)/row_sums[:, np.newaxis] # N*n_class
        return p

    def score(self, X, y):
        epsilon = 1e-7
        N = X.shape[0]
        p = self.predict_proba( X) #  N*n_class
        J = (-1./N)* np.sum(np.log(p + epsilon) * y)
        return J

    def predict(self, X, y= None):
        p = self.predict_proba(X)
        return np.argmax(p ,axis = 1)
#
a =[]
sfmax = MY_Softmax( alpha = .1, history = a, max_iter = 5001)
sfmax.fit(X_train, y_train_one_hot)

plt.plot(range(5001), a)
print sfmax.theta_

print np.mean(y_train == sfmax.predict(X_train))
print np.mean(y_valid == sfmax.predict(X_valid))

# early stopping
best_score_valid = float('inf')
history_score_valid, history_score_train = [],[]
sfmax = MY_Softmax( alpha = .1, max_iter = 1)
sfmax.fit(X_train, y_train_one_hot)
for epoch in range(10000):
    sfmax = MY_Softmax( alpha = .01, max_iter = 1, initial_theta = sfmax.theta_)
    sfmax.fit(X_train, y_train_one_hot)
    history_score_train.append(sfmax.score(X_train, y_train_one_hot))
    a = sfmax.score(X_valid, y_valid_one_hot)
    history_score_valid.append(a)
    if a < best_score_valid:
        best_score_valid = a

    else:
        print "earlys stop condition"
        print (epoch - 1)
        break


print sfmax.theta_

plt.figure()
plt.plot(range(10000), history_score_train, 'b')
plt.plot(range(10000), history_score_valid, 'r')

plt.figure()
x0, x1 = np.meshgrid(
        np.linspace(0, 8, 500).reshape(-1, 1),
        np.linspace(0, 3.5, 200).reshape(-1, 1),
        )
X_new = np.c_[x0.ravel(), x1.ravel()]
z = sfmax. predict(X_new)
zz = z.reshape(x0.shape)
plt.contourf(x0, x1, zz)

plt.plot(X_train[y_train == 0,0], X_train[y_train == 0,1],'ob')
plt.plot(X_train[y_train == 1,0], X_train[y_train == 1,1],'sg')
plt.plot(X_train[y_train == 2,0], X_train[y_train == 2,1],'^k')


plt.show()
