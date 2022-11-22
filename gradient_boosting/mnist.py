from sklearn import datasets
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV

#データの読み込み
digits = datasets.load_digits()

#訓練データとテストデータに分割
x_train, x_test, t_train, t_test = train_test_split(digits.data, digits.target, shuffle=False)


# サーチするパラメータをセット
params = {'eta': [0.01, 0.1, 1.0], 'gamma': [0, 0.1], 
                  'n_estimators': [10, 100], 'max_depth':[2, 4], 
                  'min_child_weigh': [1, 2], 'nthread': [2] }

# クラス分類用のモデルを作成
model = xgb.XGBClassifier()

# StratifiedKFoldでグリッドサーチ
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
tuned_model = GridSearchCV(estimator=model, 
                    param_grid=params, 
                    cv=skf, 
                    scoring="accuracy", 
                    n_jobs=1, 
                    verbose=3)
tuned_model.fit(x_train, t_train)


best_model = tuned_model.best_estimator_
best_params = tuned_model.best_params_

print('best parameters')
print(best_params)
print('best train score = {}'.format(best_model.score(x_train, t_train)))
print('test score by best moderl = {}'.format(best_model.score(x_test, t_test)))