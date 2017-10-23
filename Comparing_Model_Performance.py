from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
#### import for visuilization

import matplotlib.pyplot as plt


knn = KNeighborsClassifier()
iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.4)

logReg = LogisticRegression()
logReg.fit(x_train, y_train)
y_pred = logReg.predict( x_test )

knn5 = KNeighborsClassifier(n_neighbors=5)
knn1 = KNeighborsClassifier(n_neighbors=5)
knn5.fit( x_train, y_train )
knn1.fit( x_train, y_train )
y_pred5 = knn5.predict( x_test )
y_pred1 = knn1.predict( x_test )

print 'The accuracy score for K == 5 is : {}'.format( metrics.accuracy_score( y_test, y_pred5 ) )
print 'The accuracy score for logistic reg is {}'.format(metrics.accuracy_score( y_test, y_pred ) )
print 'The accuracy score for k== 1 reg is {}'.format(metrics.accuracy_score( y_test, y_pred1 ) )


### now lets try with for loop so see the behavior of K ###
k = range(1,26)
scr = []
for i in k:
     knn = KNeighborsClassifier(n_neighbors = i )
     knn.fit( x_train, y_train)
     y_p = knn.predict( x_test )
     scr.append(metrics.accuracy_score( y_test, y_p ) )
print scr
plt.plot(k, scr)
plt.xlabel('Value of K')
plt.ylabel('Scores')
plt.show()




 

