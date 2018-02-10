#encoding=UTF-8
#LDA和PCA降维
from sklearn import datasets
import numpy as np
from sklearn.preprocessing import  StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA

sc=StandardScaler()
pca=PCA(n_components=2)
iris=datasets.load_iris()
lr=LogisticRegression()

X=iris.data[:,[0,1,2,3]]
Y=iris.target

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

X_train_pca=pca.fit_transform(X_train)
X_test_pca=pca.transform(X_test)

lr.fit(X_train_pca,Y_train)

Y_pred=lr.predict(X_test_pca)

plot_decision_regions.plot_decision_regions(X_train_pca,Y_train,classifier=lr)
#plt.show()

print accuracy_score(Y_test,Y_pred)

print (Y_pred!=Y_test).sum()

lda=LDA(n_components=2)

X_train_lda=lda.fit_transform(X_train,Y_train)
X_test_lda=lda.transform(X_test)
lr.fit(X_train_lda,Y_train)
Y_pred=lr.predict(X_test_lda)

print accuracy_score(Y_test,Y_pred)