'''
sklearn tree:
    https://github.com/scikit-learn/scikit-learn/tree/master/sklearn
'''

from sklearn.datasets import load_digits
from matplotlib import pyplot as plt

#date set
print("load_digits(): ",load_digits())

#data and images
print("load_digits().data.shape: ", load_digits().data.shape)
print("load_digits().images.shape: ", load_digits().images.shape)

#first element of data and images
print("load_digits().data[0]: ", load_digits().data[0])
print("load_digits().data[0].shape: ", load_digits().data[0].shape)

print("load_digits().images[0]: ", load_digits().images[0])
print("load_digits().images[0].shape: ", load_digits().images[0].shape)

#show images
plt.imshow(load_digits().images[0], cmap="Greys")
plt.show()
plt.imshow(load_digits().data[0].reshape(8,8), cmap="Greys")
plt.show()

#target: for each images[8*8] in datasets
#there is one data[64]
#there is one number target[integer]
print("load_digits().target", load_digits().target,"load_digits().target.shape", load_digits().target.shape)
print("load_digits().target[0]:",  load_digits().target[0])
plt.imshow(load_digits().images[0], cmap="Greys")
plt.show()
plt.imshow(load_digits().data[0].reshape(8,8), cmap="Greys")
plt.show()
