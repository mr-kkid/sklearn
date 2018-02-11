from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X,Y=make_blobs(n_samples=150,n_features=2,centers=3,cluster_std=0.5,shuffle=True,random_state=0)

#plt.scatter(X[:,0],X[:,1],c='blue',marker='o',s=50)
#plt.grid()
#plt.show()

for i in range(10):

    kmeans=KMeans(n_clusters=3,init='random',n_init=i+1,max_iter=300,tol=1e-4,random_state=None)

    Y_pred=kmeans.fit_predict(X)

    plt.scatter(X[Y_pred==0,0],X[Y_pred==0,1],c='lightgreen',marker='s',s=50,label='cluster 1')

    plt.scatter(X[Y_pred==1,0],X[Y_pred==1,1],c='orange',marker='o',s=50,label='cluster 2')

    plt.scatter(X[Y_pred==2,0],X[Y_pred==2,1],c='lightblue',marker='v',s=50,label='cluster 3')

    plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=250,marker='*',c='red',label='centroids')

    plt.legend()

    plt.grid()
    plt.show()

