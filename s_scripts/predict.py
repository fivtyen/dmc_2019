from datetime import datetime
import os.path
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

# start time measurement
start = datetime.now()

# read test data
df = pd.read_csv('../data/test.csv', '|')

# load the model
model_file = '../data/model/clf.sav'
clf = joblib.load(model_file)

# create result file
result_file = '../data/predictions.csv'
with open(result_file, 'w') as f:
    f.write('0_confidence,1_confidence,predicted_class')

# make predictions
prob = list(clf.predict_proba(df))
prob = [list(p) for p in prob]

# format output
for p in prob:
    if p[0] > p[1]:
        p.append(0)
    else:
        p.append(1)

# save output
with open(result_file, 'a') as f:
    for p in prob:
        if p[-1] != p[-2]:
            f.write('{},{},{}\n'.format(*p))

# print runtime
end = datetime.now()
print('FINISHED! TOTAL RUNTIME: {}'.format(end-start))