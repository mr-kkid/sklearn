from sklearn import datasets
import numpy as np
from sklearn.preprocessing import  StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

sc=StandardScaler()
iris=datasets.load_iris()

X=iris.data[:,[0,1,2,3]]
Y=iris.target

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

sc.fit(X_train)

X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test);


ppn=Perceptron(n_iter=4,eta0=0.1,random_state=0)
ppn.fit(X_train_std,Y_train)

Y_pred=ppn.predict(X_test_std)



print (Y_pred!=Y_test).sum()

print accuracy_score(Y_test,Y_pred)
