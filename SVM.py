
## All the imports
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.svm import SVC

#data

digits = load_digits()

data = digits.data
target = digits.target
x, y = data[:-1], target[:-1]
clf = SVC()
clf.fit(x,y)
print clf
print clf.predict(data[-1])
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
