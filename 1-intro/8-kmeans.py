from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
'''
make sample
'''
X, y=make_blobs(n_samples=100, n_features=2, centers=3)
plt.scatter(X[:,0],X[:,1])
plt.show()


'''
kmeans
'''
kmeans_obj=KMeans(n_clusters=3)

#train
kmeans_obj.fit(X)

#labels:
labels=kmeans_obj.predict(X)
print(labels)

plt.scatter(X[:,0], X[:,1], c=labels)
plt.show()


'''
score
'''
print(y, labels)
print(adjusted_rand_score(y , labels)*100)

while True:
    '''
    perdict kmeans?!?!?!?
    '''
    new_X, new_y=make_blobs(n_samples=50, n_features=2, centers=4)
    plt.scatter(new_X[:,0],new_X[:,1])
    plt.show()

    perdict_new_sample_lables=kmeans_obj.predict(new_X)
    print(perdict_new_sample_lables)
    plt.scatter(X[:,0], X[:,1], c=labels)
    plt.scatter(new_X[:,0], new_X[:,1], c=perdict_new_sample_lables, marker="x")
    plt.show()
