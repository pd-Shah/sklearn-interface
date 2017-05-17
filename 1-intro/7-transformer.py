from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#load dataset
dataset=load_boston()

#split test and train
X_train, X_test, y_train, y_test= train_test_split(dataset.data, dataset.target)

# mean and standard diviation
print("X_train", X_train)
print("mean: ", X_train.mean(axis=0))
print("standard diviation: ", X_train.std(axis=0))

print("X_train.shape: ", X_train.shape)
print("X_train.std(axis=0).shape: ", X_train.std(axis=0).shape)
print("X_train.mean(axis=0).shape ", X_train.mean(axis=0).shape)


# StandardScaler transformer
#make an objeect
transformer_obj=StandardScaler()

#train
transformer_obj.fit(X_train)
print(transformer_obj)

#transform
transformed=transformer_obj.transform(X_train)
print("transformed", transformed)
print("transformed mean: ", transformed.mean(axis=0))
print("transformed standard diviation: ", transformed.std(axis=0))

print(transformed.max())
print(transformed.min())
print(X_train.max())

print(\
"min max transformer"
)
min_max_obj=MinMaxScaler()
transformed=(min_max_obj.fit_transform(X_train))
print(transformed)
print(transformed.max())
print(transformed.min())
