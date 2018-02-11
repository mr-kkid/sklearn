#encoding=UTF-8
#K-fold and cross_validation and precision and recall
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
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold,cross_val_score
from sklearn.metrics import f1_score,recall_score,precision_score

iris=datasets.load_iris()


X=iris.data[:,[0,1,2,3]]
Y=iris.target

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

pipline_lr=Pipeline([('scl',StandardScaler()),('pca',PCA(n_components=2)),('cls',LogisticRegression(random_state=1))])

kfold=StratifiedKFold(y=Y_train,n_folds=10,random_state=1)

scores=[]
for k,(train,test) in enumerate(kfold):
    pipline_lr.fit(X_train[train],Y_train[train])
    score=pipline_lr.score(X_train[test],Y_train[test])
    scores.append(score)
    print ('Fold %d:%.3f'%(k+1,score))

scores=cross_val_score(estimator=pipline_lr,X=X_train,y=Y_train,n_jobs=2,cv=10)

print scores

print '\n'
#encoding=UTF-8
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
from sklearn.pipeline import Pipeline


iris=datasets.load_iris()


X=iris.data[:,[0,1,2,3]]
Y=iris.target

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

print np.mean(scores)

lr=LogisticRegression()
lr.fit(X_train,Y_train)
Y_pred=lr.predict(X_test)

print '-----------------------------\n'

print precision_score(y_true=Y_test,y_pred=Y_pred,sample_weight=None)

print recall_score(y_true=Y_test,y_pred=Y_pred,sample_weight=None)

print f1_score(y_true=Y_test,y_pred=Y_pred,sample_weight=None)