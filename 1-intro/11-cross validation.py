from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt

'''
KNeighborsClassifier
'''
X, y=load_iris(return_X_y=True)
X_train, X_test, y_train, y_test=train_test_split(X, y)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
pred=knn.predict(X_test)

plt.scatter(X_train[:,0], X_train[:,1], c=y_train)
plt.scatter(X_test[:,0], X_test[:,1], c=pred, marker="x")
plt.show()


'''
cross_val_score
'''
from sklearn.cross_validation import cross_val_score
score=cross_val_score(knn, X, y)
print(score)

_=score.mean()
print(_)


'''
kfold
'''
from sklearn.cross_validation import KFold
kf=KFold(n=25, n_folds=5, shuffle=False)
for index, item in enumerate(kf, start=1):
    print(index, ": ", "train:", item[0], "test:", item[1])

kf=KFold(n=X.shape[0], n_folds=5, shuffle=False)
for index, item in enumerate(kf, start=1):
    print(index, ": ", "train:", item[0], "test:", item[1])
_=cross_val_score(knn, X, y, cv=kf)
print(_.mean())


'''
ploting
'''
import numpy as npy
from sklearn.cross_validation import ShuffleSplit

def show(cv, n):
    masks=[]
    for train, test in cv:
        mask=npy.zeros(n, dtype=bool)
        mask[test]=1
        masks.append(mask)
    plt.matshow(masks)
    plt.show()

kf=KFold(n=25, n_folds=5, shuffle=False)
show(kf, 25)

kf=KFold(n=25, n_folds=5, shuffle=True)
show(kf, 25)

shp=ShuffleSplit(n=25, n_iter=10, test_size=0.2)
show(shp, 25)
