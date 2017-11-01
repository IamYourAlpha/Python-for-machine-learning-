import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import binarize
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
col_names = ['pregrant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(url, header=None, names=col_names)
feature_col  = ['pregrant', 'insulin', 'bmi', 'age']
x = pima[feature_col]
y = pima.label
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=10)

LogReg = LogisticRegression()
LogReg.fit(x_train,y_train)

#sns.pairplot(pima, x_vars = feature_col, y_vars = 'label', size=7, aspect=0.7, kind='scatter')
#plt.show()
y_pred = LogReg.predict(x_test)
per =  metrics.accuracy_score(y_test, y_pred)

#print y_pred

#print 'True:', y_test.values[0:25]
#print 'Predi', y_pred[0:25]


ConMat =  metrics.confusion_matrix(y_test, y_pred)

TP = ConMat [1,1]
TN = ConMat [0,0]
FP = ConMat [0,1]
FN = ConMat [1,0]
#
#print "predicting the classification accuracy"
#print per
#print (TP+TN)/float(TP+TN+FP+FN)

#print " Predicting the misclasfication "

#print 1-per
#print (FN+FP)/float(TP+TN+FP+FN)


#print ('Precdicting the sensitivity or recall : %f')%(TP/float(TP+FN))
#print ('Precdicting the sensitivity or recall'), metrics.recall_score(y_test, y_pred)

print LogReg.predict(x_test)[0:25]
print LogReg.predict_proba(x_test)[0:25, :]
y_pred_prob =  LogReg.predict_proba(x_test)[:, 1]
plt.hist(y_pred_prob)
plt.xlim(0,1)
plt.show()
y_pred_class = binarize(y_pred_prob, 0.3)
print y_pred_class

