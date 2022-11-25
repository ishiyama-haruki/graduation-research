from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#データの読み込み
digits = datasets.load_digits()

#訓練データとテストデータに分割
x_train, x_test, t_train, t_test = train_test_split(digits.data, digits.target, shuffle=False)


model = RandomForestClassifier()
params = {
    'n_estimators': [10, 20, 30, 50, 100, 300],
    'criterion': ["gini", "entropy"],
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': [10, 20, 30, 40, 50]
}


n_iter = 100
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
# {'criterion': 'entropy', 'max_depth': None, 'min_samples_split': 10, 'n_estimators': 300}
# best train score = 0.9992576095025983
# test score by best moderl = 0.9222222222222223