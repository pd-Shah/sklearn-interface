import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve
from mpl_toolkits.mplot3d import Axes3D

'''
make and plot 3d
'''
X, y = make_s_curve(n_samples=1000)
ax = plt.axes(projection='3d')

ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=y)
ax.view_init(10, -60)
plt.show()

'''
linear
PCA decomposition
'''
from sklearn.decomposition import PCA

pca=PCA(n_components=2).fit(X)
transform=pca.transform(X)
plt.scatter(transform[:,0], transform[:,1], c=y)
plt.show()


'''
nonlinear
isomap
'''
from sklearn.manifold import Isomap

iso=Isomap(n_neighbors=20, n_components=2)
iso.fit(X)
transform=iso.transform(X)
plt.scatter(transform[:,0], transform[:,1], c=y)
plt.show()
