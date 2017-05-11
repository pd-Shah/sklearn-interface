from sklearn.datasets import load_digits
from matplotlib import pyplot as plt

#load_digits by return_X_y=True
dataset=load_digits(n_class=10, return_X_y=True)
print("dataset: ", dataset)
print("len dataset: ", len(dataset))


#len items of dataset
#64=> 8*8
print("dataset[0].shape: ", dataset[0].shape)
print("dataset[1].shape: ", dataset[1].shape)

# X-> data, y-> target
X, y= dataset

#splite train and test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y)

print("X_train:",  X_train, "\nX_train.shape:", X_train.shape)
print("X_test:",   X_test,  "\nX_test.shape:",  X_test.shape)
print("y_train:",  y_train, "\ny_train.shape:", y_train.shape)
print("y_test:",   y_test,  "\ny_test.shape:",  y_test.shape)
