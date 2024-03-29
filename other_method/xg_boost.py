from scripts import sample_data
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
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

model = lgb.LGBMClassifier(objective='multiclass')

# params = {
#     'learning_rate' :  [1e-3],#[1e-3, 1e-2, 0.1, 1],
#     'num_leaves': [16],#[16, 32, 64, 128, 256, 512, 1024],
#     'n_estimators': [500],#[500, 1000],
#     'max_depth': [10],#[10]
# }
params = {
    'num_leaves': [3, 4, 5, 6, 7, 8, 9, 10],
    'reg_alpha': [0, 1, 2, 3, 4, 5,10, 100],
    'reg_lambda': [10, 15, 18, 20, 21, 22, 23, 25, 27, 29]
}


cv = 5
tuned_model = GridSearchCV(
    estimator=model, 
    param_grid=params,
    cv = cv,
    verbose = 100
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