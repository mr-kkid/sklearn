from sklearn import datasets
import numpy as np
from sklearn.preprocessing import  StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import plot_decision_regions
import matplotlib.pyplot as plt

sc=StandardScaler()
iris=datasets.load_iris()

X=iris.data[:,[2,3]]
Y=iris.target

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

sc.fit(X_train)

X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test);


SVM=SVC(kernel='linear',C=1.0,random_state=0)
SVM.fit(X_train_std,Y_train)

Y_pred=SVM.predict(X_test_std)

print  accuracy_score(Y_test,Y_pred)

SVM=SVC(kernel='rbf',gamma=2.0,C=1.0,random_state=0)
SVM.fit(X_train_std,Y_train)

Y_pred=SVM.predict(X_test_std)

X_combined_std=np.vstack((X_train_std,X_test_std))
Y_combined=np.hstack((Y_train,Y_test))
plot_decision_regions.plot_decision_regions(X=X_combined_std,Y=Y_combined,classifier=SVM,test_idx=range(105,150))
plt.show()

print  accuracy_score(Y_test,Y_pred)