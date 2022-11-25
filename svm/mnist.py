from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


#データの読み込み
digits = datasets.load_digits()

#訓練データとテストデータに分割
x_train, x_test, t_train, t_test = train_test_split(digits.data, digits.target, shuffle=False)


model = SVC()
params = [
    {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ['linear']},
    {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
    {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
    {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
    ]


cv = 5
tuned_model = GridSearchCV(
    estimator=model, 
    param_grid=params,
    cv = cv
)
tuned_model.fit(x_train, t_train)

best_model = tuned_model.best_estimator_
best_params = tuned_model.best_params_

print('best parameters')
print(best_params)
print('best train score = {}'.format(best_model.score(x_train, t_train)))
print('test score by best moderl = {}'.format(best_model.score(x_test, t_test)))

# best parameters
# {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}
# best train score = 1.0
# test score by best moderl = 0.9688888888888889