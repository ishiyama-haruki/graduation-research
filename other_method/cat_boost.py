from scripts import sample_data
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
import numpy as np
import sys
import time

start_time = time.time()

dataset = sys.argv[1]

if dataset == 'mnist':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_mnist()
elif dataset == 'usps':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_usps()
elif dataset == 'covtype':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_covtype()
elif dataset == 'ijcnn1':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_ijcnn1()
elif dataset == 'letter':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_letter()
elif dataset == 'cifar10':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_cifar10()
elif dataset == 'dna':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_dna()
elif dataset == 'aloi':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_aloi()
elif dataset == 'sector':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_sector()
elif dataset == 'shuttle':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_shuttle()
elif dataset == 'susy':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_susy()

print("dataset {} is loaded".format(dataset))
print('--------------------------------------')
sys.stdout.flush() # 明示的にflush

model = CatBoostClassifier()

params = {
    'depth': [6,7,8,9,10],
    'learning_rate': [0.01,0.02,0.03,0.04],
    'iterations': [100, 200, 300],
    'logging_level':['Silent']
}

scorer = make_scorer(accuracy_score)

cv = 5
tuned_model = GridSearchCV(
    estimator=model, 
    param_grid=params,
    cv = cv,
    scoring=scorer,
)

tuned_model.fit(X, Y)

best_model = tuned_model.best_estimator_
best_params = tuned_model.best_params_

end_time = time.time()

print('best parameters')
print(best_params)
print('best train score = {}'.format(best_model.score(X, Y)))
print('test score by best model = {}'.format(best_model.score(Xt, Yt)))
print('time: {}'.format(end_time- start_time))