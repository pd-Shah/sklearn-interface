from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

x,y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y)
pca = PCA(n_components=2)
clf_svm = SVC(kernel='linear')

###
# Pipeline
###
pipeline = Pipeline([('pca', pca), ('clf_svm', clf_svm)])
pipeline.fit(x_train, y_train)
pipeline.score(x_train, y_train)
pipeline.score(x_test, y_test)
pipeline.set_params(pca__n_components=3, clf_svm__kernel='rbf').fit(x_train, y_train)

###
# GridsearchCV
###
from sklearn.model_selection import GridSearchCV
pipeline.set_params(clf_svm__kernel='rbf')
estimator = GridSearchCV(pipeline, dict(pca__n_components=[2,3,4],
                                       clf_svm__C=[0.1,10,100],
                                       clf_svm__gamma=[0.1,10,100]))
estimator.fit(x_train, y_train)
print("best_estimator_: ", estimator.best_estimator_)
print("best_params_: ", estimator.best_params_)
print("best_score_: ", estimator.best_score_)
