import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score
url =  'http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv'
data = pd.read_csv(url, index_col=0)

feature_cols = [ 'TV', 'radio', 'newspaper' ]
x = data[ feature_cols ]
y = data.sales

linReg = LinearRegression()
scores = cross_val_score(linReg, x, y, cv  = 10, scoring='mean_squared_error')
#print scores*-1
rmse = np.sqrt(scores*-1)
print rmse.mean()

print 'Now without the newspaper'
feature_cols = ['TV', 'radio']
x = data[ feature_cols ]
scores = cross_val_score(linReg, x, y, cv  = 10, scoring='mean_squared_error')
#print scores*-1
print np.sqrt(scores*-1).mean()

