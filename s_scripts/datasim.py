import os.path
import pandas as pd
import random

from datetime import datetime

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import CondensedNearestNeighbour, RandomUnderSampler
from imblearn.combine import SMOTEENN

# number of simulation runs
N = 50
# results filename
results_file = '../data/datasimres_2.csv'
# set random seed to current time
random.seed(datetime.now())

# prepare data
df = pd.read_csv('../data/train.csv', '|')
X = df.drop('fraud', axis=1)
y = df['fraud']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create results file
with open(results_file, 'w') as f:
    f.write('model;accuracy;precision;recall;roc\n')

# random forest parameters
params = {'bootstrap': True,
          'max_depth': 80,
          'max_features': 'sqrt',
          'min_samples_leaf': 2,
          'min_samples_split': 5,
          'n_estimators': 1000 #,
          # 'random_state': 42
}


def get_model(params=params):
    clf = RandomForestClassifier(bootstrap=params['bootstrap'],
                              max_depth=params['max_depth'],
                              max_features=params['max_features'],
                              min_samples_leaf=params['min_samples_leaf'],
                              min_samples_split=params['min_samples_split'],
                              n_estimators=params['n_estimators'] #,
                            #   random_state=params['random_state']
                            )
    return clf


# def train_and_measure(model, data_model_name, X_res, y_res, X_test=X_test, y_test=y_test):
def train_and_measure(model, data_model_name, X_res, y_res, X_test, y_test):
    model.fit(X_res, y_res)
    pred = model.predict(X_test)
    acc = round(model.score(X_test, y_test), 4)
    prec = round(precision_score(y_test, pred), 4)
    rec = round(recall_score(y_test, pred), 4)
    roc = round(roc_auc_score(y_test, pred), 4)

    with open(results_file, 'a') as f:
        f.write('{};{};{};{};{}\n'.format(data_model_name, acc, prec, rec, roc))


# SMOTE
print('SMOTE')
for i in range(N):
    clf = get_model()
    sm = SMOTE()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    train_and_measure(clf, 'smote', X_res, y_res, X_test, y_test)

# ADASYN
print('ADASYN')
for i in range(N):
    clf = get_model()
    ad = ADASYN()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_res, y_res = ad.fit_resample(X_train, y_train)
    train_and_measure(clf, 'adasyn', X_res, y_res, X_test, y_test)

# ROS
print('ROS')
for i in range(N):
    clf = get_model()
    ros = RandomOverSampler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_res, y_res = ros.fit_resample(X_train, y_train)
    train_and_measure(clf, 'ros', X_res, y_res, X_test, y_test)

# SMOTE + ENN
print('SMOTEENN')
for i in range(N):
    clf = get_model()
    smnn = SMOTEENN()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_res, y_res = smnn.fit_resample(X_train, y_train)
    train_and_measure(clf, 'smoteenn', X_res, y_res, X_test, y_test)

# CNN
print('CNN')
for i in range(N):
    clf = get_model()
    cnn = CondensedNearestNeighbour()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_res, y_res = cnn.fit_resample(X_train, y_train)
    train_and_measure(clf, 'cnn', X_res, y_res, X_test, y_test)

# RUS
print('RUS')
for i in range(N):
    clf = get_model()
    rus = RandomUnderSampler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    train_and_measure(clf, 'rus', X_res, y_res, X_test, y_test)
