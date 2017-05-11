from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split

from sklearn.datasets import load_digits

# load dataset
# X data, y target
X,y=load_digits(return_X_y=True)

X_train, X_test, y_train, y_test= train_test_split(X,y)

#make object from model
svm_obj=LinearSVC()

#run model on data:
result=svm_obj.fit(X_train, y_train)
print(result)

#make prediction
predict=svm_obj.predict(X_test)

#compare
print(predict)
print(y_test)

#predict score
print(svm_obj.score(X_test, y_test))
