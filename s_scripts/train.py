import os.path
import pandas as pd

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from imblearn.over_sampling import RandomOverSampler

# read training data
df = pd.read_csv('../data/train.csv', '|')

# prepare data for training
# testing is not necessary therefore all data is used as trainig set
X = df.drop('fraud', axis=1)
y = df['fraud']
ro = RandomOverSampler()
X_res, y_res = ro.fit_resample(X, y)

# prepare model - parameters from tuning
parameters = {'n_estimators': 1000,
              'min_samples_split': 2,
              'min_samples_leaf': 1,
              'max_features': 'auto',
              'max_depth': None,
              'bootstrap': False}

clf = RandomForestClassifier(n_estimators=parameters['n_estimators'],
                             min_samples_split=parameters['min_samples_split'],
                             min_samples_leaf=parameters['min_samples_leaf'],
                             max_features=parameters['max_features'],
                             max_depth=parameters['max_depth'],
                             bootstrap=parameters['bootstrap'])

# train the model
clf.fit(X_res, y_res)

# save the trained model
model_dir = '../data/model'
model_file = os.path.join(model_dir, 'clf.sav')

if not os.path.exists(model_file):
    os.mkdir(model_dir)

joblib.dump(clf, model_file)

print('Model trained and saved!')
