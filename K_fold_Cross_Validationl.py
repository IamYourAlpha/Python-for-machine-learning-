from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test  = train_test_split(x, y, random_state = 2)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print metrics.accuracy_score(y_test, y_pred)

knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, x, y, cv=10, scoring = 'accuracy')
print scores 
print scores.mean()
k_range = range(1,31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print k_scores

plt.plot(k_range, k_scores)
plt.xlabel('value of K for knn')
plt.ylabel('scores')
plt.show()

LogReg = LogisticRegression()
knn20 = KNeighborsClassifier(n_neighbors=20)
print 'printing the scores for logistic Regression'
scoresLog = cross_val_score(LogReg, x, y, cv=10, scoring='accuracy')
print scoresLog.mean()
scoresknn = cross_val_score(knn20, x, y, cv=10, scoring='accuracy')
print 'printing the scors for k nearest neigbor'
print scoresknn.mean()



