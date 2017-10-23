import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression 
# read the data

data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

#sns.pairplot(data, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', size=7, aspect=0.7, kind='reg')
#plt.show()

feature_cols = ['TV', 'radio', 'newspaper']
x = data[feature_cols]
y = data['sales']

x_train, y_train, x_test, y_test = train_test_split(x, y, random_state = 1)
print x_train
print y_train 
linReg = LinearRegression()
linReg.fit(data['TV'], data['sales'])
print linReg 
