from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
#load dataset
dataset=load_boston()
print(dataset.keys())

#split train and test
X_train, X_test, y_train, y_test= train_test_split(dataset.data, dataset.target)

#make model object
ridge_obj=Ridge()

#train
ridge_obj.fit(X_train, y_train)

#prediction
ridge_obj.predict(X_test)

# evaluation
print(ridge_obj.score(X_test, y_test))

#mean_squared_error evaluation
print(mean_squared_error(y_test, ridge_obj.predict(X_test)))
