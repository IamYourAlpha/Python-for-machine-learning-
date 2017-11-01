from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import RandomizedSearchCV
import matplotlib.pyplot as plt
k_range = range(1,31)

param_grid = dict(n_neighbors=k_range)
print param_grid

iris = load_iris()
x = iris.data
y = iris.target

knn = KNeighborsClassifier()

grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(x,y)

grid_mean_scores = [ result.mean_validation_score for result in grid.grid_scores_]

plt.plot(k_range, grid_mean_scores)
plt.xlabel('value of k')
plt.ylabel('value of mean scores')
#plt.show()

print grid.best_score_
print grid.best_estimator_
print grid.best_params_
weight_options = ['uniform', 'distance']
params_dist = dict(n_neighbors=k_range, weights=weight_options)
rand = RandomizedSearchCV(knn, params_dist, cv=10, scoring='accuracy', n_iter=10)
rand.fit(x,y)
print rand.best_score_


