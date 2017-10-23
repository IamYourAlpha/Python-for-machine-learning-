from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
#################################
 ## Loading the dataset##
iris = load_iris()
type(iris)
data    =  iris.data
x       =  data   
y =  iris.target
###############################

## Loading the KNeigboursClassfier ##
knn     =   KNeighborsClassifier(n_neighbors=1)
knn.fit(x,y)
print 'Printing by K nearest neigbors'
print iris.target_names[knn.predict([3,4,5,2])]
##---------------------##


## Loading the logistic Regression ##
print 'Predicting by logistic Regression '
LogReg = LogisticRegression()
LogReg.fit(x,y)
print iris.target_names[LogReg.predict([3,4,5,2])]
