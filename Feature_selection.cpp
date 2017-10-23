import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

url = 'http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv'
data = pd.read_csv(url, index_col = 0 )
print data.head
