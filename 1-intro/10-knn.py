from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

'''
load datasets
'''
X, y=load_iris(return_X_y=True)

print(X.shape)
print(y.shape)


'''
splite train test
'''
X_train, X_test, y_train, y_test=train_test_split(X, y)
print(X_train.shape, y_train.shape)

'''
make classifier
'''
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

'''
predict
'''
pred=knn.predict(X_test)

'''
ploting
'''
plt.scatter(X_train[:,0], X_train[:,1], c=y_train)
plt.scatter(X_test[:,0], X_test[:,1], c=pred, marker="x")
plt.show()

'''
validation

	               Predicted

                Cat	Dog	Rabbit
Actual	        Cat	5	3	0
                Dog	2	3	1
            Rabbit	0	2	11


'''

_=confusion_matrix(y_test, pred)
print(_)
