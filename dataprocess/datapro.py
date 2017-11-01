import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
x_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('Y_train.csv')

# for testing 

x_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('Y_test.csv')

#print x_train.head()

#print x_train[x_train.dtypes[ (x_train.dtypes=="float64") | (x_train.dtypes=="int64")].index.values].hist(figsize=[11,11])
#plt.show()

knn = KNeighborsClassifier(n_neighbors=5)
feature = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
knn.fit(x_train[feature], y_train)

from sklearn.metrics import accuracy_score

y_pred = knn.predict(x_test[feature])
scre = accuracy_score(y_test, y_pred)
print"the accuracy score is ",  scre 

print y_train.Target.value_counts()/y_train.Target.count()
print y_train.Target.count()

print y_test.Target.value_counts()/y_test.Target.count()

