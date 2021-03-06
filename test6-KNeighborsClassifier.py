#encoding=UTF-8
#K相邻算法
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.cross_validation import train_test_split
import numpy as np
import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=5,p=1,metric='minkowski')
sc=StandardScaler()
iris=datasets.load_iris()

X=iris.data[:,[2,3]]
Y=iris.target

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

sc.fit(X_train)

X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test);

knn.fit(X_train_std,Y_train)

Y_pred=knn.predict(X_test_std)

X_combined_std=np.vstack((X_train_std,X_test_std))
Y_combined=np.hstack((Y_train,Y_test))
plot_decision_regions.plot_decision_regions(X=X_combined_std,Y=Y_combined,classifier=knn,test_idx=range(105,150))
plt.show()


print accuracy_score(Y_test,Y_pred)