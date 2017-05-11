from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits

from sklearn.ensemble import RandomForestClassifier

#make RandomForestClassifier object
rfc_obj=RandomForestClassifier()

#make train and test
X,y=load_digits(return_X_y=True)
X_train, X_test, y_train, y_test= train_test_split(X, y)

#training
rfc_obj.fit(X_train, y_train)

rfc_obj.predict(X_test)

print(rfc_obj.score(X_test, y_test))
